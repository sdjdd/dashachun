import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { z } from "zod";
import { ApiError, api } from "./api";
import { copy } from "./copy";

export type Role = "user" | "admin";

export const credentialsSchema = z.object({
  username: z
    .string()
    .min(1, copy.auth.validation.usernameRequired)
    .regex(/^[A-Za-z0-9_-]{3,32}$/, copy.auth.validation.usernameFormat),
  password: z
    .string()
    .min(1, copy.auth.validation.passwordRequired)
    .min(8, copy.auth.validation.passwordLength)
    .max(128, copy.auth.validation.passwordLength),
});

export type Credentials = z.infer<typeof credentialsSchema>;

export interface User {
  id: number;
  username: string;
  role: Role;
}

async function fetchSession(): Promise<User | null> {
  return api<User>("/api/auth/me").catch((err) => {
    if (err instanceof ApiError && err.status === 401) return null;
    throw err;
  });
}

export const sessionQuery = {
  queryKey: ["session"] as const,
  queryFn: fetchSession,
  staleTime: "static" as const,
  retry: false,
};

export function useSession() {
  return useQuery(sessionQuery);
}

export function useLogin() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (creds: Credentials) =>
      api<User>("/api/auth/login", { method: "POST", body: JSON.stringify(creds) }),
    onSuccess: (user) => queryClient.setQueryData(sessionQuery.queryKey, user),
  });
}

export function useRegister() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (creds: Credentials) =>
      api<User>("/api/auth/register", { method: "POST", body: JSON.stringify(creds) }),
    onSuccess: (user) => queryClient.setQueryData(sessionQuery.queryKey, user),
  });
}

export function useLogout() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => api<void>("/api/auth/logout", { method: "POST" }),
    onSuccess: () => queryClient.setQueryData(sessionQuery.queryKey, null),
  });
}

export function describeAuthError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 401) return copy.auth.errors.invalidCredentials;
    if (err.status === 409) return copy.auth.errors.usernameTaken;
    if (err.status === 429) return copy.auth.errors.rateLimited(err.retryAfter ?? 60);
    return err.message;
  }
  return copy.auth.errors.network;
}
