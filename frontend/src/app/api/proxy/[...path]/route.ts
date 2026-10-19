import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';
export const maxDuration = 120; // Allow sufficient time for large PDF uploads

// Strict allowlist of paths required by the Next.js frontend UI
const ALLOWED_PATH_PATTERNS: RegExp[] = [
  /^fonts\/?$/,
  /^fonts\/upload$/,
  /^config$/,
  /^pdfs$/,
  /^pdfs\/upload$/,
  /^pdfs\/upload-bulk$/,
  /^pdfs\/[a-zA-Z0-9_-]+$/,
  /^pdfs\/[a-zA-Z0-9_-]+\/status$/,
  /^pdfs\/[a-zA-Z0-9_-]+\/result$/,
  /^pdfs\/[a-zA-Z0-9_-]+\/file$/,
  /^pdfs\/[a-zA-Z0-9_-]+\/math-status$/,
  /^pdfs\/[a-zA-Z0-9_-]+\/convert-equations$/,
  /^pdfs\/[a-zA-Z0-9_-]+\/revalidate$/,
  /^pdfs\/[a-zA-Z0-9_-]+\/rescan\/[0-9]+$/,
  /^pdfs\/[a-zA-Z0-9_-]+\/questions\/[0-9]+\/image$/,
  /^pdfs\/[a-zA-Z0-9_-]+\/questions\/[0-9]+\/image_info$/,
  /^fix_question_with_ai$/,
  /^verify_question_with_ai$/,
  /^review-flags$/,
  /^review-flags\/[0-9]+\/resolve$/,
  /^review-flags\/stats$/,
  /^cache\/stats$/,
];

function getBackendBase(): string {
  if (process.env.BACKEND_URL) return process.env.BACKEND_URL;
  if (process.env.INTERNAL_BACKEND_URL) return process.env.INTERNAL_BACKEND_URL;
  return 'http://localhost:8000';
}

async function handleProxy(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  const { path } = await context.params;
  const pathStr = path ? path.join('/') : '';

  // 1. Path Allowlist check: reject any path outside UI requirements
  const isAllowed = ALLOWED_PATH_PATTERNS.some((pattern) => pattern.test(pathStr));
  if (!isAllowed) {
    return NextResponse.json(
      { detail: `Forbidden: Path '${pathStr}' is not in proxy allowlist.` },
      { status: 403 }
    );
  }

  // 2. Extract client API key from x-api-key header or active session cookie
  const sessionCookie = request.cookies.get('qscan_session')?.value;
  const callerApiKey = request.headers.get('x-api-key');
  const backendApiKey = callerApiKey || (sessionCookie && sessionCookie.startsWith('qscan_live_') ? sessionCookie : null);

  if (!backendApiKey) {
    return NextResponse.json(
      { detail: 'Unauthorized: Valid API key required. Please log in with your client key.' },
      { status: 401 }
    );
  }

  const search = request.nextUrl.search || '';
  const backendBase = getBackendBase();
  const targetUrl = `${backendBase.replace(/\/+$/, '')}/${pathStr}${search}`;

  const headers = new Headers();
  const forwardHeaders = ['content-type', 'accept', 'content-length', 'user-agent'];
  for (const h of forwardHeaders) {
    const val = request.headers.get(h);
    if (val) {
      headers.set(h, val);
    }
  }
  headers.set('X-API-Key', backendApiKey);

  const fetchOptions: RequestInit = {
    method: request.method,
    headers,
  };

  if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(request.method)) {
    try {
      const bodyBuffer = await request.arrayBuffer();
      if (bodyBuffer.byteLength > 0) {
        fetchOptions.body = bodyBuffer;
      }
    } catch {
      // Empty body
    }
  }

  try {
    let backendRes = await fetch(targetUrl, fetchOptions).catch(async (primaryErr) => {
      if (backendBase.includes('backend:8000')) {
        const fallbackUrl = `http://localhost:8000/${pathStr}${search}`;
        return await fetch(fallbackUrl, fetchOptions);
      }
      throw primaryErr;
    });

    const resHeaders = new Headers();
    backendRes.headers.forEach((val, key) => {
      if (!['content-encoding', 'transfer-encoding', 'connection'].includes(key.toLowerCase())) {
        resHeaders.set(key, val);
      }
    });

    const resBody = await backendRes.arrayBuffer();
    return new NextResponse(resBody, {
      status: backendRes.status,
      statusText: backendRes.statusText,
      headers: resHeaders,
    });
  } catch (err: any) {
    return NextResponse.json(
      { detail: `Frontend server proxy error: failed to connect to backend at ${targetUrl}: ${err.message}` },
      { status: 502 }
    );
  }
}

export const GET = handleProxy;
export const POST = handleProxy;
export const PUT = handleProxy;
export const DELETE = handleProxy;
export const PATCH = handleProxy;
