import { copy } from "./copy";

export class ApiError extends Error {
  readonly status: number;
  readonly retryAfter?: number;

  constructor(status: number, message: string, retryAfter?: number) {
    super(message);
    this.status = status;
    this.retryAfter = retryAfter;
  }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: init?.body ? { "content-type": "application/json", ...init.headers } : init?.headers,
  });
  if (!res.ok) {
    const body = (await res.json().catch(() => null)) as { error?: string } | null;
    throw new ApiError(
      res.status,
      body?.error ?? copy.common.requestFailed(res.status),
      Number(res.headers.get("retry-after")) || undefined,
    );
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}
