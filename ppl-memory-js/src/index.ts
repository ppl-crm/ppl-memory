/**
 * ppl as a memory backend for the Vercel AI SDK.
 *
 * ppl (https://withppl.com) is a personal CRM. `createPplTools` returns a set
 * of AI SDK tools that give an agent long-term memory about people: memories
 * live in the user's ppl account as notes on contact timelines, so they
 * survive across sessions and are inspectable in the ppl web app.
 *
 * Usage:
 *
 *   import { generateText } from 'ai';
 *   import { createPplTools } from 'ppl-memory';
 *
 *   const { text } = await generateText({
 *     model: myModel,
 *     tools: createPplTools({ apiToken: process.env.PPL_API_TOKEN }),
 *     prompt: 'What does Jane like to drink?',
 *   });
 */

import { tool } from 'ai';
import { z } from 'zod';

export interface PplToolsOptions {
  /** ppl API token (falls back to the PPL_API_TOKEN env var). */
  apiToken?: string;
  /** ppl base URL, defaults to https://withppl.com. */
  baseUrl?: string;
}

export interface PplRecallHit {
  id: unknown;
  text: string;
  score?: number;
  contact?: string;
}

const DEFAULT_BASE_URL = 'https://withppl.com';

function resolveToken(apiToken?: string): string {
  const token = apiToken ?? process.env.PPL_API_TOKEN;
  if (!token) {
    throw new Error(
      'No API token. Pass apiToken= or set the PPL_API_TOKEN environment variable ' +
        '(create one at https://withppl.com/settings/agents).'
    );
  }
  return token;
}

async function pplRequest(
  baseUrl: string,
  apiToken: string,
  method: string,
  path: string,
  body?: Record<string, unknown>
): Promise<unknown> {
  const headers: Record<string, string> = {
    Accept: 'application/json',
    Authorization: `Bearer ${apiToken}`,
  };
  let payload: string | undefined;
  if (body !== undefined) {
    payload = JSON.stringify(body);
    headers['Content-Type'] = 'application/json';
  }
  const res = await fetch(`${baseUrl}${path}`, { method, headers, body: payload });
  if (!res.ok) {
    throw new Error(`ppl API ${res.status} on ${method} ${path}`);
  }
  const text = await res.text();
  return text ? JSON.parse(text) : null;
}

/** Laravel resources wrap collections in {"data": [...]}; unwrap for convenience. */
function unwrap(payload: unknown): any {
  if (
    payload !== null &&
    typeof payload === 'object' &&
    !Array.isArray(payload) &&
    'data' in payload &&
    Object.keys(payload).length <= 3
  ) {
    return (payload as { data: unknown }).data;
  }
  return payload;
}

function asArray(value: unknown): any[] {
  if (value === null || value === undefined) return [];
  return Array.isArray(value) ? value : [value];
}

function hitText(r: any): string {
  return r.title ?? r.text ?? r.body ?? JSON.stringify(r);
}

/**
 * Build the ppl tool set for the Vercel AI SDK. Returns tools keyed by name:
 * `pplRemember`, `pplRecall`, `pplBriefing`, `pplFindContact`.
 */
export function createPplTools(options: PplToolsOptions = {}) {
  const apiToken = resolveToken(options.apiToken);
  const baseUrl = (options.baseUrl ?? DEFAULT_BASE_URL).replace(/\/$/, '');

  return {
    pplRemember: tool({
      description:
        'Store a fact about a person in ppl. The fact is saved as a note on ' +
        "their timeline and recalled in later sessions.",
      inputSchema: z.object({
        contactId: z
          .union([z.number(), z.string()])
          .describe('The ppl contact id (use pplFindContact to resolve a name).'),
        fact: z.string().describe('The fact to remember, in plain text.'),
      }),
      execute: async ({ contactId, fact }) => {
        const note = unwrap(
          await pplRequest(baseUrl, apiToken, 'POST', '/api/notes', {
            contact_id: contactId,
            body: fact,
          })
        ) as { id?: unknown } | null;
        return { id: note?.id ?? null, message: `Remembered (note ${note?.id ?? 'unknown'}).` };
      },
    }),

    pplRecall: tool({
      description:
        "Search ppl's memory for facts about people. Returns ranked results, " +
        'best matches first.',
      inputSchema: z.object({
        query: z.string().describe('The question or topic to search for.'),
        limit: z.number().int().positive().optional().default(10),
      }),
      execute: async ({ query, limit }): Promise<PplRecallHit[]> => {
        const results = unwrap(
          await pplRequest(baseUrl, apiToken, 'POST', '/api/agent/ask', {
            question: query,
            limit,
          })
        );
        return asArray(results).map((r) => ({
          id: r.id,
          text: hitText(r),
          score: r.score,
          contact: r.contact ?? r.contact_name,
        }));
      },
    }),

    pplBriefing: tool({
      description:
        'Get the ppl morning briefing: birthdays, people to reconnect with, ' +
        'pending tasks, and suggested actions.',
      inputSchema: z.object({}),
      execute: async () => {
        return (
          unwrap(await pplRequest(baseUrl, apiToken, 'GET', '/api/agent/briefing')) ?? {}
        );
      },
    }),

    pplFindContact: tool({
      description:
        'Find the best-matching ppl contact by name. Returns the contact, or ' +
        'null when there is no match.',
      inputSchema: z.object({
        name: z.string().describe('The name to search for.'),
      }),
      execute: async ({ name }) => {
        const params = new URLSearchParams({ q: name, types: 'contacts', limit: '3' });
        const results = unwrap(
          await pplRequest(baseUrl, apiToken, 'GET', `/api/search/semantic?${params}`)
        );
        const list = asArray(results);
        return list[0] ?? null;
      },
    }),
  };
}

export type PplTools = ReturnType<typeof createPplTools>;
