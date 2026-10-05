import { beforeEach, describe, expect, it, vi } from 'vitest';
import { createPplTools } from './index.js';

const ok = (payload: unknown) =>
  new Response(JSON.stringify(payload), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });

const err = (status: number) => new Response('boom', { status });

function mockFetch(handler: (url: string, init: any) => Response) {
  const calls: Array<{ url: string; init: any }> = [];
  const fn = vi.fn(async (url: string, init: any) => {
    calls.push({ url, init });
    return handler(url, init);
  });
  vi.stubGlobal('fetch', fn);
  return calls;
}

beforeEach(() => {
  vi.unstubAllGlobals();
  process.env.PPL_API_TOKEN = 'test-token';
});

describe('createPplTools', () => {
  it('throws without a token', () => {
    delete process.env.PPL_API_TOKEN;
    expect(() => createPplTools()).toThrow(/No API token/);
  });

  it('sends the Bearer token on every call', async () => {
    const calls = mockFetch(() => ok({ data: {} }));
    const tools = createPplTools();
    await tools.pplBriefing.execute!({}, { toolCallId: 'x', messages: [] } as any);
    expect(calls[0].init.headers.Authorization).toBe('Bearer test-token');
  });

  it('pplRemember posts a note', async () => {
    const calls = mockFetch(() => ok({ data: { id: 7, body: 'likes tea' } }));
    const tools = createPplTools();
    const result = await tools.pplRemember.execute!(
      { contactId: 42, fact: 'likes tea' },
      { toolCallId: 'x', messages: [] } as any
    );
    expect(calls[0].url).toBe('https://withppl.com/api/notes');
    expect(calls[0].init.method).toBe('POST');
    expect(JSON.parse(calls[0].init.body)).toEqual({ contact_id: 42, body: 'likes tea' });
    expect(result).toEqual({ id: 7, message: 'Remembered (note 7).' });
  });

  it('pplRecall posts the question and maps hits', async () => {
    mockFetch(() =>
      ok({ data: [{ id: 7, title: 'Jane likes tea', score: 0.9, contact_name: 'Jane Smith' }] })
    );
    const tools = createPplTools();
    const hits = await tools.pplRecall.execute!(
      { query: 'What does Jane drink?', limit: 10 },
      { toolCallId: 'x', messages: [] } as any
    );
    expect(hits).toEqual([
      { id: 7, text: 'Jane likes tea', score: 0.9, contact: 'Jane Smith' },
    ]);
  });

  it('pplRecall returns an empty list when there are no hits', async () => {
    mockFetch(() => ok({ data: [] }));
    const tools = createPplTools();
    const hits = await tools.pplRecall.execute!(
      { query: 'nothing', limit: 10 },
      { toolCallId: 'x', messages: [] } as any
    );
    expect(hits).toEqual([]);
  });

  it('pplBriefing gets the briefing', async () => {
    const calls = mockFetch(() => ok({ data: { birthdays: ['Jane Smith'] } }));
    const tools = createPplTools();
    const briefing = await tools.pplBriefing.execute!(
      {},
      { toolCallId: 'x', messages: [] } as any
    );
    expect(calls[0].url).toBe('https://withppl.com/api/agent/briefing');
    expect(briefing).toEqual({ birthdays: ['Jane Smith'] });
  });

  it('pplFindContact returns the best match', async () => {
    mockFetch(() => ok({ data: [{ id: 42, name: 'Jane Smith' }] }));
    const tools = createPplTools();
    const contact = await tools.pplFindContact.execute!(
      { name: 'Jane' },
      { toolCallId: 'x', messages: [] } as any
    );
    expect(contact).toEqual({ id: 42, name: 'Jane Smith' });
  });

  it('pplFindContact returns null when there is no match', async () => {
    mockFetch(() => ok({ data: [] }));
    const tools = createPplTools();
    const contact = await tools.pplFindContact.execute!(
      { name: 'Nobody' },
      { toolCallId: 'x', messages: [] } as any
    );
    expect(contact).toBeNull();
  });

  it('surfaces API errors', async () => {
    mockFetch(() => err(401));
    const tools = createPplTools();
    await expect(
      tools.pplBriefing.execute!({}, { toolCallId: 'x', messages: [] } as any)
    ).rejects.toThrow(/ppl API 401/);
  });

  it('respects a custom baseUrl and explicit token', async () => {
    const calls = mockFetch(() => ok({ data: {} }));
    const tools = createPplTools({ apiToken: 'explicit', baseUrl: 'https://example.test/' });
    await tools.pplBriefing.execute!({}, { toolCallId: 'x', messages: [] } as any);
    expect(calls[0].url).toBe('https://example.test/api/agent/briefing');
    expect(calls[0].init.headers.Authorization).toBe('Bearer explicit');
  });
});
