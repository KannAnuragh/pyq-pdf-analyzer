import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

function getBackendBase(): string {
  if (process.env.BACKEND_URL) return process.env.BACKEND_URL;
  if (process.env.INTERNAL_BACKEND_URL) return process.env.INTERNAL_BACKEND_URL;
  return 'http://localhost:8000';
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const rawApiKey = (body.apiKey || body.password || body.key || '').trim();

    if (!rawApiKey) {
      return NextResponse.json({ detail: 'Please enter a valid API key.' }, { status: 400 });
    }

    // Strictly validate against Backend API using X-API-Key header.
    // No hardcoded master passwords or backdoors allowed.
    try {
      const backendBase = getBackendBase().replace(/\/+$/, '');
      const res = await fetch(`${backendBase}/`, {
        method: 'GET',
        headers: {
          'X-API-Key': rawApiKey,
        },
      });

      if (!res.ok) {
        return NextResponse.json(
          { detail: 'Invalid or revoked API key. Please check your credentials.' },
          { status: 401 }
        );
      }
    } catch (err) {
      console.error('Backend validation check failed:', err);
      return NextResponse.json(
        { detail: 'Unable to reach backend authentication service.' },
        { status: 503 }
      );
    }

    const response = NextResponse.json({ success: true, message: 'Authenticated successfully' });
    response.cookies.set('qscan_session', rawApiKey, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      path: '/',
      maxAge: 60 * 60 * 24 * 7, // 7 days
    });

    return response;
  } catch {
    return NextResponse.json({ detail: 'Invalid login request format.' }, { status: 400 });
  }
}
