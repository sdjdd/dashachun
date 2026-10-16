import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { z } from "zod";
import { ApiError, api } from "./api";
import { copy } from "./copy";

export interface DeviceSession {
  id: string;
  ip: string;
  user_agent: string;
  created_at: string;
  last_active_at: string;
  expires_at: string;
  current: boolean;
}

export function useSessions() {
  return useQuery({
    queryKey: ["auth", "sessions"] as const,
    queryFn: () => api<DeviceSession[]>("/api/auth/sessions"),
  });
}

export const changePasswordSchema = z
  .object({
    current_password: z.string().min(1, copy.account.validation.currentRequired),
    new_password: z
      .string()
      .min(8, copy.account.validation.passwordLength)
      .max(128, copy.account.validation.passwordLength),
    confirm_password: z.string().min(1, copy.account.validation.confirmRequired),
  })
  .refine((values) => values.new_password === values.confirm_password, {
    path: ["confirm_password"],
    message: copy.account.validation.confirmMismatch,
  });

export type ChangePasswordValues = z.infer<typeof changePasswordSchema>;

export function useChangePassword() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (values: ChangePasswordValues) =>
      api<void>("/api/auth/change-password", {
        method: "POST",
        body: JSON.stringify({
          current_password: values.current_password,
          new_password: values.new_password,
        }),
      }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["auth", "sessions"] }),
  });
}

export function useRevokeSession() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api<void>(`/api/auth/sessions/${id}`, { method: "DELETE" }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["auth", "sessions"] }),
  });
}

export function describePasswordError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 401) return copy.account.errors.wrongCurrentPassword;
    if (err.status === 429) return copy.auth.errors.rateLimited(err.retryAfter ?? 60);
    return err.message;
  }
  return copy.common.network;
}

export function describeSessionError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 404) return copy.account.sessions.errors.notFound;
    if (err.status === 429) return copy.auth.errors.rateLimited(err.retryAfter ?? 60);
    return err.message;
  }
  return copy.common.network;
}
