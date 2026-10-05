# RUNBOOK.md — Deploying and checking the site

The site is static: no database, no secrets, no environment variables. A
deploy can't break a schema, but it can still break in ways CI doesn't
see, so every deploy follows the same three steps. They are adapted from
transformer-explainer's RUNBOOK §7.

## 1. Deploy from a clean export

The Vercel project (`llm-architectures-explained`) is not on Vercel's Git
integration. Deploy with the logged-in Vercel CLI from a clean export of
`HEAD`, so nothing untracked (caches, `node_modules`, local files) is
uploaded:

```bash
rm -rf /tmp/lae-deploy && mkdir /tmp/lae-deploy
git archive HEAD | tar -x -C /tmp/lae-deploy
cp -r .vercel /tmp/lae-deploy/
(cd /tmp/lae-deploy && vercel deploy --yes)          # preview
(cd /tmp/lae-deploy && vercel deploy --prod --yes)   # production
vercel ls llm-architectures-explained | head             # newest must be ● Ready
```

Test a preview first. Previews are protected by Vercel Authentication; to
smoke-check one, create a protection-bypass token in the project settings
and pass it as `VERCEL_BYPASS` (never commit or print it).

## 2. Smoke-check

```bash
pnpm smoke https://llm-architectures-explained.vercel.app
# a protected preview:
VERCEL_BYPASS=… pnpm smoke https://<preview-url>
```

It fetches every page (every model page included) and fails on any non-200
(redirects included) or on a page without the content that proves it
rendered real data: each model page's name and provenance chips, its
diagram when the model has dimensions, and its labelled estimates when it
has any. It also fetches `/data/specs.json`, the compare tool's data, checks
it holds every model, and runs the cost model on DeepSeek-V3 from it (671B
expected). Then open `/compare` in a browser and change the context slider
and the estimates toggle: the tool runs client-side, which the smoke check
can't see.

## 3. Read the logs

```bash
vercel logs --environment production --since 15m --no-branch --expand
```

A static site should log almost nothing. On the Hobby plan the CLI only
reaches back about an hour; the dashboard's Logs view keeps more.

## Why e2e runs under `--no-experimental-require-module`

Plain Node 20.19+ / 22.12+ can `require()` an ES module; Vercel's function
loader can't. transformer-explainer shipped a comment renderer that passed
every local and CI test and returned 500 on Vercel for that reason
(2026-10-04). This site has no server functions, but CI runs the e2e
server under the flag anyway, so the class of bug can't arrive unnoticed if
one is added.
