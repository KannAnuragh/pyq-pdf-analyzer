import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  const session = request.cookies.get('qscan_session')?.value;
  const authenticated = Boolean(session && session.trim().length > 0);
  return NextResponse.json({ authenticated });
}
