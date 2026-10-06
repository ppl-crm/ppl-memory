# ppl-memory (TypeScript)

ppl ([withppl.com](https://withppl.com)) as a memory backend for the
[Vercel AI SDK](https://sdk.vercel.ai).

ppl is a personal CRM. `createPplTools` returns AI SDK tools that give an
agent long-term memory about people. Memories live in the user's ppl account
as notes on contact timelines, so they survive across sessions and are
inspectable in the ppl web app.

## Install

Not published to npm yet. Build and install from source:

```bash
git clone https://github.com/ppl-crm/ppl-memory.git
cd ppl-memory/ppl-memory-js
npm install
npm run build
# then, from your project:
npm install /path/to/ppl-memory/ppl-memory-js
```

## Setup

Create an API token at
[withppl.com/settings/agents/setup](https://withppl.com/settings/agents/setup) and export
it:

```bash
export PPL_API_TOKEN="your-token"
```

## Usage

```ts
import { generateText } from 'ai';
import { createPplTools } from 'ppl-memory';

const { text } = await generateText({
  model: myModel,
  tools: createPplTools(), // reads PPL_API_TOKEN
  prompt: 'What does Jane like to drink?',
});
```

Or pass the token explicitly: `createPplTools({ apiToken: '...' })`.

## Tools

| Tool | What it does |
|---|---|
| `pplRemember` | Store a fact about a contact (needs `contactId`; use `pplFindContact` to resolve a name). |
| `pplRecall` | Ranked retrieval over everything ppl knows (`query`, `limit`). |
| `pplBriefing` | Morning briefing: birthdays, reconnects, pending tasks, suggested actions. |
| `pplFindContact` | Best-matching contact by name, or `null`. |

## Develop

```bash
npm install
npm test    # vitest
npm run build  # tsc -> dist/
```

## License

MIT
