# Content Credentials trust settings

- `C2PA-TRUST-LIST.pem`: the official C2PA trust list of certificate authorities allowed to sign Content Credentials, from [c2pa-org/conformance-public](https://github.com/c2pa-org/conformance-public/tree/main/trust-list) (CC BY 4.0).
- `eku.cfg`: the certificate purposes (extended key usages) accepted for signing.

To refresh the list:

```bash
curl -L -o C2PA-TRUST-LIST.pem https://raw.githubusercontent.com/c2pa-org/conformance-public/main/trust-list/C2PA-TRUST-LIST.pem
```
