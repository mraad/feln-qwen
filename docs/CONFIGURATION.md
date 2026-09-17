# Privacy and local configuration

The current tracked files use neutral SSH aliases, configurable paths, and a
private install-time repository setting. Personal account names, home paths,
machine addresses, SSH key filenames, and the dependency repository owner were
removed. Unrelated directory listings were omitted from hardware inventories.

Changed text logs have a **Sanitized evidence** header. Their host/account labels
and paths are placeholders, not literal command output. Hardware measurements,
benchmark JSON, model identifiers, revision pins, and artifact hashes are unchanged.
Trailing spaces in text evidence were trimmed for the initial public commit.
The Studio screenshot contains the application and petroleum-domain example data;
its image and checksum are unchanged. Catalog layer names, company operators,
well names, and geographic coordinates remain domain data.

## Recommended environment variables

Start with [`.env.example`](../.env.example). Create a private `.env` only if one
does not already exist, fill in the real paths/repository, and load it explicitly:

```sh
cp -n .env.example .env
chmod 600 .env
# Edit .env for this host before loading it. Source only your own trusted file.
set -a
. ./.env
set +a
```

Use separate values on each host. `.env` and `.env.*` are ignored except for the
sanitized example. No new dotenv loader is required.

| Variable | Purpose | Where used |
|---|---|---|
| `FELN_APP_REPO` | Absolute path to the application reference checkout | Mac preparation and verification commands |
| `NORTHSEA_SOURCE` | Absolute path to read-only source data | Mac preparation and database-copy commands |
| `TRAIN_VENV` | Existing verified training Python environment | RTX train/evaluate/export commands |
| `LLAMA_CPP_DIR` | Existing llama.cpp checkout/build | RTX conversion and quantization commands |
| `LAYERS_JSON_REPOSITORY` | Credential-free `git+https://…` or `git+ssh://…` repository URL | Both serving requirements files; pip/uv expand `${LAYERS_JSON_REPOSITORY}` |
| `FELN_ROOT` | App bundle/database root | Already read by `serving/app.py` |
| `FELN_MODEL_URL` | Local model endpoint | Already read by `serving/app.py` |

The dependency revision remains pinned to
`32d6d4ea57975a05552f2ade840cf2085ff25ed1` in both requirements files. Set
`LAYERS_JSON_REPOSITORY` in the installation shell before running pip or uv; it
must not include the revision suffix. The example URL is intentionally unusable.
Keep authentication in an SSH agent or credential helper, not in the URL or Git.
Package-manager metadata and `pip freeze` can reveal the resolved repository URL;
redact them before adding future environment snapshots to Git.

Do **not** move model revisions, checksums, dataset split seeds, LoRA parameters,
or benchmark settings into secret configuration. They are reproducibility data.

## SSH and systemd settings

Keep real host addresses, login names, and key paths in your private
`~/.ssh/config`, using `rtx-training` and `orin` as aliases. Documentation uses
`ssh rtx-training`, `ssh orin`, and those same aliases for SCP. The aliases are
labels, not DNS names; configure them before running commands. SSH configuration
is a better fit for these settings than additional application environment variables.

The checked-in units now use the neutral example account `feln`. **This does not
rename the existing deployment account or create a new account.** After copying
these templates into place, and before starting them, set the intended existing
unprivileged account in local unit drop-ins using `sudo systemctl edit UNIT`, for each of the three project units:

```ini
[Service]
User=YOUR_EXISTING_SERVICE_ACCOUNT
```

Replace that placeholder locally. Ensure the account can read the model, bundle,
database and venv, write its logs, and access its DuckDB extension cache. Keep the
drop-ins outside Git. `User=` does not expand shell environment variables; do not
replace it with `${FELN_SERVICE_USER}`. No remote units were changed by this cleanup.

For the API's existing runtime variables, a local drop-in can add:

```ini
[Service]
EnvironmentFile=/etc/feln-qwen.env
```

That host-local file can contain `FELN_ROOT=…` and `FELN_MODEL_URL=…`. Changing
`FELN_ROOT` alone does not relocate unit `WorkingDirectory`, executable, log paths,
or Studio/model arguments; update those local unit settings together if relocating
the deployment. Fixed ports and experiment directory names are not personal data.
These are configuration instructions, not changes applied to the stopped services.

## Remaining private material

This public repository starts with a fresh history containing the sanitized files.
Earlier development commits and their author/committer metadata remain only in the
original private repository; they were not imported here. The public GitHub account
name is necessarily visible in repository ownership and CODEOWNERS. Historical
cleanup evidence describes the original private checkout before this publication.

The original private workspace retains ignored `runs/` source snapshots and archives
with upstream identifiers and paths. Those artifacts are absent from this public
repository. They were not rewritten because that would invalidate the recorded
source checksums and provenance. They must remain private and must not be added
with `git add -f` or included in a public workspace archive. Move them to a private
artifact store outside the checkout before sharing the whole directory. Original
NorthSea data, sibling repositories, and deployed remote files are unchanged.

The scan covered tracked text, common credential patterns, address/email patterns,
the screenshot, and known identifiers in ignored artifacts. It is not a guarantee
that arbitrary upstream datasets or historical archives contain no personal data.
The [cleanup check results](evidence/privacy-cleanup.txt) record the validation
and the material deliberately left unchanged.
