"""Provenance checks on the uploaded file: Content Credentials (C2PA), AI labels in the metadata and camera EXIF."""

import io
import json
import re
from pathlib import Path

import c2pa

TRUST = Path(__file__).parent / "trust"

# IPTC digital source types (https://cv.iptc.org/newscodes/digitalsourcetype/)
AI_TYPES = {"trainedAlgorithmicMedia", "compositeWithTrainedAlgorithmicMedia", "algorithmicMedia", "compositeSynthetic"}
CAMERA_TYPES = {"digitalCapture", "computationalCapture"}
MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "HEIF": "image/heif",
        "AVIF": "image/avif", "TIFF": "image/tiff", "GIF": "image/gif"}
GENERATOR_NAMES = re.compile(r"midjourney|dall.?e|stable diffusion|firefly|imagen|gemini|chatgpt|openai|grok|ideogram|"
                             r"leonardo|flux|novelai|comfyui", re.IGNORECASE)
EXIF_SOFTWARE, EXIF_MAKE, EXIF_MODEL = 0x0131, 0x010F, 0x0110


def _context():
    settings = c2pa.Settings.from_dict({
        # no network access: the check must also work offline
        "verify": {"verify_trust": True, "remote_manifest_fetch": False, "ocsp_fetch": False},
        "trust": {"trust_anchors": (TRUST / "C2PA-TRUST-LIST.pem").read_text(), "trust_config": (TRUST / "eku.cfg").read_text()},
    })
    return c2pa.Context(settings)


CONTEXT = _context()


def _source_type(uri):
    return uri.rstrip("/").rsplit("/", 1)[-1] if uri else None


def content_credentials(data, image_format):
    """Reads and verifies the C2PA manifest store, or returns None if the file has none."""
    mime = MIME.get(image_format)
    if mime is None:
        return None
    try:
        with c2pa.Reader(mime, io.BytesIO(data), context=CONTEXT) as reader:
            store = json.loads(reader.json())
    except Exception:
        return None
    manifests = store.get("manifests", {})
    active = manifests.get(store.get("active_manifest"), {})

    # actions of the active manifest and of its ingredients (e.g. an AI image later edited in another program)
    actions = [action for manifest in manifests.values() for assertion in manifest.get("assertions", [])
               if assertion.get("label", "").startswith("c2pa.actions") for action in assertion.get("data", {}).get("actions", [])]
    types = {_source_type(a.get("digitalSourceType")) for a in actions} - {None}
    created = next((a for a in actions if a.get("action") == "c2pa.created"), {})
    agent = created.get("softwareAgent") or next((a["softwareAgent"] for a in actions if a.get("softwareAgent")), None)
    if isinstance(agent, dict):
        agent = " ".join(str(agent[k]) for k in ("name", "version") if agent.get(k))
    generator = (active.get("claim_generator_info") or [{}])[0].get("name") or active.get("claim_generator")
    signature = active.get("signature_info", {})
    state = store.get("validation_state", "Invalid" if store.get("validation_status") else "Valid")

    return {
        "state": state,  # Trusted: valid signature from a certificate on the C2PA trust list; Valid: valid but signer not listed; Invalid
        "tool": agent or generator,
        "signer": signature.get("issuer") or signature.get("common_name"),
        "signed_at": signature.get("time"),
        "source_types": sorted(types),
        "ai": bool(types & AI_TYPES),
        "camera": bool(types & CAMERA_TYPES) and not types & AI_TYPES,
        "problems": [s.get("explanation") or s.get("code") for s in store.get("validation_status", [])],
    }


def metadata_label(data, img):
    """Unsigned hints that a generator made the file: IPTC/XMP source type, generator settings or software name."""
    match = re.search(rb"digitalsourcetype/(\w+)", data[:4_000_000], re.IGNORECASE)
    if match and match.group(1).decode() in AI_TYPES:
        return f"IPTC digital source type '{match.group(1).decode()}'"
    if "parameters" in img.info and "Steps:" in str(img.info["parameters"]):
        return "Stable Diffusion generation settings"
    if "workflow" in img.info or ("prompt" in img.info and str(img.info["prompt"]).lstrip().startswith("{")):
        return "ComfyUI workflow"
    try:
        software = str(img.getexif().get(EXIF_SOFTWARE) or img.info.get("Software") or "")
    except Exception:
        software = ""
    if GENERATOR_NAMES.search(software):
        return f"software '{software.strip()}'"
    return None


def camera_exif(img):
    try:
        exif = img.getexif()
    except Exception:
        return None
    make, model = (str(exif.get(tag) or "").strip("\x00 ") for tag in (EXIF_MAKE, EXIF_MODEL))
    if not model:
        return make or None
    return model if model.lower().startswith(make.lower()) else f"{make} {model}".strip()


def check(data, img):
    """Provenance of an uploaded file. `img` must be the PIL image as opened, before any conversion."""
    credentials = content_credentials(data, img.format)
    label = None if credentials else metadata_label(data, img)
    state = credentials["state"] if credentials else None
    # an AI claim needs a valid signature; a camera claim also needs a signer on the trust list,
    # because anyone could sign a false "captured by a camera" claim with their own certificate
    if state in ("Trusted", "Valid") and credentials["ai"]:
        decision = "ai"
    elif state == "Trusted" and credentials["camera"]:
        decision = "camera"
    elif label:
        decision = "ai_label"
    else:
        decision = None
    return {"credentials": credentials, "label": label, "camera": camera_exif(img), "decision": decision}
