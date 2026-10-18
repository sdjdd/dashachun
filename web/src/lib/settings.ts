import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { z } from "zod";
import { ApiError, api } from "./api";
import { copy } from "./copy";

export type ProviderType = "asr" | "llm" | "tts";

export interface ProviderRow {
  id: number;
  type: ProviderType;
  name: string;
  kind: string;
  config: Record<string, unknown>;
  selected: boolean;
  created_at: string;
  updated_at: string;
}

export interface FieldSpec {
  key: string;
  required: boolean;
  secret?: boolean;
  number?: boolean;
  min?: number;
  placeholderKey?: keyof typeof copy.settings.providers.placeholders;
  hintKey?: keyof typeof copy.settings.providers.hints;
}

export const PROVIDER_KINDS: Record<ProviderType, Array<{ kind: string; fields: FieldSpec[] }>> = {
  asr: [
    {
      kind: "volcengine",
      fields: [
        {
          key: "base_url",
          required: true,
          hintKey: "asrBaseUrl",
        },
        { key: "api_key", required: true, secret: true },
        { key: "resource_id", required: true },
      ],
    },
  ],
  llm: [
    {
      kind: "openai",
      fields: [
        { key: "base_url", required: true },
        { key: "api_key", required: true, secret: true },
        { key: "model", required: true },
        { key: "max_tokens", required: false, number: true, min: 1, placeholderKey: "maxTokens" },
      ],
    },
  ],
  tts: [
    {
      kind: "volcengine",
      fields: [
        {
          key: "base_url",
          required: true,
          hintKey: "ttsBaseUrl",
        },
        { key: "api_key", required: true, secret: true },
        { key: "resource_id", required: true },
        { key: "speaker", required: true },
      ],
    },
  ],
};

export function providerKinds(type: ProviderType): Array<{ kind: string }> {
  return PROVIDER_KINDS[type] ?? [];
}

export function providerFields(type: ProviderType, kind: string): FieldSpec[] {
  return PROVIDER_KINDS[type]?.find((entry) => entry.kind === kind)?.fields ?? [];
}

export type ProviderFormValues = { name: string; kind: string } & Record<string, string>;

export function providerFormValues(
  type: ProviderType,
  kind: string,
  row?: ProviderRow,
): ProviderFormValues {
  const values: ProviderFormValues = { name: row?.name ?? "", kind };
  for (const spec of providerFields(type, kind)) {
    const raw = row?.config?.[spec.key];
    values[spec.key] = raw === undefined || raw === null ? "" : String(raw);
  }
  return values;
}

export function providerConfigPayload(
  type: ProviderType,
  kind: string,
  values: ProviderFormValues,
): Record<string, unknown> {
  const config: Record<string, unknown> = {};
  for (const spec of providerFields(type, kind)) {
    const raw = (values[spec.key] ?? "").trim();
    if (raw === "") continue;
    config[spec.key] = spec.number ? Number(raw) : raw;
  }
  return config;
}

export function useProviders() {
  return useQuery({
    queryKey: ["providers"] as const,
    queryFn: () => api<ProviderRow[]>("/api/providers"),
  });
}

export function useCreateProvider() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (input: {
      type: ProviderType;
      name: string;
      kind: string;
      config: Record<string, unknown>;
    }) =>
      api<ProviderRow>("/api/providers", {
        method: "POST",
        body: JSON.stringify(input),
      }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["providers"] }),
  });
}

export function useUpdateProvider() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (input: {
      id: number;
      name: string;
      kind: string;
      config: Record<string, unknown>;
    }) =>
      api<ProviderRow>(`/api/providers/${input.id}`, {
        method: "PUT",
        body: JSON.stringify({ name: input.name, kind: input.kind, config: input.config }),
      }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["providers"] }),
  });
}

export function useDeleteProvider() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api<void>(`/api/providers/${id}`, { method: "DELETE" }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["providers"] }),
  });
}

export function useActivateProvider() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: number) =>
      api<ProviderRow>(`/api/providers/${id}/activate`, { method: "POST" }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["providers"] }),
  });
}

export interface OtaSettings {
  websocket_url: string | null;
  timezone_offset: number;
}

export function useOtaSettings() {
  return useQuery({
    queryKey: ["settings", "ota"] as const,
    queryFn: () => api<OtaSettings>("/api/settings/ota"),
  });
}

export const otaSchema = z.object({
  websocket_url: z
    .string()
    .trim()
    .refine((value) => value === "" || /^wss?:\/\//.test(value), copy.settings.ota.validation.url),
  timezone_offset: z
    .string()
    .trim()
    .refine((value) => /^-?\d+$/.test(value), copy.settings.ota.validation.offsetInteger)
    .refine((value) => {
      const offset = Number(value);
      return offset >= -720 && offset <= 840;
    }, copy.settings.ota.validation.offsetRange),
});

export type OtaFormValues = z.infer<typeof otaSchema>;

export function useSaveOtaSettings() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (input: { websocket_url: string | null; timezone_offset: number }) =>
      api<OtaSettings>("/api/settings/ota", { method: "PUT", body: JSON.stringify(input) }),
    onSuccess: (saved) => queryClient.setQueryData(["settings", "ota"], saved),
  });
}

export interface AudioCaptureSettings {
  enabled: boolean;
  bucket: string;
}

export function useAudioCapture() {
  return useQuery({
    queryKey: ["settings", "audio-capture"] as const,
    queryFn: () => api<AudioCaptureSettings>("/api/settings/audio-capture"),
  });
}

export const audioCaptureSchema = z
  .object({
    enabled: z.boolean(),
    bucket: z.string().trim(),
  })
  .refine((values) => !values.enabled || values.bucket !== "", {
    path: ["bucket"],
    message: copy.settings.capture.validation.bucketRequired,
  });

export type AudioCaptureFormValues = z.infer<typeof audioCaptureSchema>;

export function useSaveAudioCapture() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (input: AudioCaptureSettings) =>
      api<AudioCaptureSettings>("/api/settings/audio-capture", {
        method: "PUT",
        body: JSON.stringify(input),
      }),
    onSuccess: (saved) => queryClient.setQueryData(["settings", "audio-capture"], saved),
  });
}

export interface VoiceprintSettings {
  match_threshold: number;
}

export function useVoiceprintSettings() {
  return useQuery({
    queryKey: ["settings", "voiceprint"] as const,
    queryFn: () => api<VoiceprintSettings>("/api/settings/voiceprint"),
  });
}

export const voiceprintSchema = z.object({
  match_threshold: z
    .string()
    .trim()
    .refine(
      (value) => value !== "" && Number.isFinite(Number(value)),
      copy.settings.voiceprint.validation.thresholdNumber,
    )
    .refine((value) => {
      const threshold = Number(value);
      return threshold >= 0 && threshold <= 1;
    }, copy.settings.voiceprint.validation.thresholdRange),
});

export type VoiceprintFormValues = z.infer<typeof voiceprintSchema>;

export function useSaveVoiceprintSettings() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (input: VoiceprintSettings) =>
      api<VoiceprintSettings>("/api/settings/voiceprint", {
        method: "PUT",
        body: JSON.stringify(input),
      }),
    onSuccess: (saved) => queryClient.setQueryData(["settings", "voiceprint"], saved),
  });
}

const ERROR_PREFIXES = ["Bad request: ", "Conflict: ", "Service unavailable: "];

function plainMessage(message: string): string {
  for (const prefix of ERROR_PREFIXES) {
    if (message.startsWith(prefix)) return message.slice(prefix.length);
  }
  return message;
}

export function describeProviderError(err: unknown, opts?: { conflict?: string }): string {
  if (err instanceof ApiError) {
    if (err.status === 409 && opts?.conflict) return opts.conflict;
    if (err.status === 404) return copy.settings.providers.errors.notFound;
    return plainMessage(err.message);
  }
  return copy.common.network;
}

export function describeSettingsError(err: unknown): string {
  if (err instanceof ApiError) return plainMessage(err.message);
  return copy.common.network;
}
