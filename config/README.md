# Model configuration

`llm.json` holds the non-secret provider settings used by `scripts/redesign/clients.py`.
Model and endpoint changes belong here; the adapter deliberately rejects an
unreviewed model or endpoint until its capabilities and pricing bounds are updated.
The provider is Baseten; the configured model is the agreed GLM-5.3-Flash.

Put only the API key in the repository-root `.env`, preserving existing entries:

```dotenv
BASETEN_API_KEY=your-key-here
```

`api_key_env` names the environment variable; it is not the key itself. Keep keys
out of this directory. Model and URL values do not need duplicate `.env` entries.
The guarded adapter reads these settings and resolves the key inside its client
process without printing it. Editing this configuration does not require reading
`.env`. Jev uses `TYPESAFE_API_KEY` in the same runtime-only way.

GLM text/image and Jev typed requests were live-validated in autonomous run
`20260917T215500Z`. Legacy runners remain historical and are not the guarded
entry points. `redesign_runtime.json` contains explicit research defaults and the
shared session cap; `jev_questions_v1.json` contains the initial question bank.
Use the run's handover instructions and existing ledger when resuming. A restart
does not reset spending, deadlines, sample boundaries or blocked discovery routes.

The base URL and exact model identifier were checked against Baseten's
[official GLM-5.3-Flash model page](https://www.baseten.co/library/glm-53-flash/)
on 2026-09-17. The URL is the OpenAI-compatible API base, so it does not include
`/chat/completions`.
