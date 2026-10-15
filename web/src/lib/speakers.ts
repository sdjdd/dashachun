import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { z } from "zod";
import { ApiError, api } from "./api";
import { copy } from "./copy";

export interface Speaker {
  id: number;
  agent_id: number;
  name: string;
  description: string;
  sample_count: number;
  created_at: string;
  updated_at: string;
}

export const createSpeakerSchema = z.object({
  name: z
    .string()
    .min(1, copy.agents.speakers.validation.nameRequired)
    .max(64, copy.agents.speakers.validation.nameLength),
  description: z.string().max(200, copy.agents.speakers.validation.descriptionLength),
  message_ids: z
    .array(z.number().int().positive())
    .min(1, copy.agents.speakers.errors.pickRange)
    .max(10, copy.agents.speakers.errors.pickRange),
});

export type CreateSpeakerInput = z.infer<typeof createSpeakerSchema>;

export const updateSpeakerSchema = z.object({
  name: z
    .string()
    .min(1, copy.agents.speakers.validation.nameRequired)
    .max(64, copy.agents.speakers.validation.nameLength),
  description: z.string().max(200, copy.agents.speakers.validation.descriptionLength),
});

export type UpdateSpeakerInput = z.infer<typeof updateSpeakerSchema>;

function speakersQuery(agentId: number) {
  return {
    queryKey: ["agents", agentId, "speakers"] as const,
    queryFn: () => api<Speaker[]>(`/api/agents/${agentId}/speakers`),
  };
}

export function useSpeakers(agentId: number) {
  return useQuery(speakersQuery(agentId));
}

export function useEnrollSpeaker(agentId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (input: CreateSpeakerInput) =>
      api<Speaker>(`/api/agents/${agentId}/speakers`, {
        method: "POST",
        body: JSON.stringify({
          name: input.name,
          ...(input.description ? { description: input.description } : {}),
          message_ids: input.message_ids,
        }),
      }),
    onSuccess: (speaker) => {
      queryClient.setQueryData<Speaker[]>(speakersQuery(agentId).queryKey, (old) =>
        old?.some((s) => s.name === speaker.name)
          ? old.map((s) => (s.name === speaker.name ? speaker : s))
          : [...(old ?? []), speaker],
      );
    },
  });
}

export function useUpdateSpeaker(agentId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ speakerId, input }: { speakerId: number; input: UpdateSpeakerInput }) =>
      api<Speaker>(`/api/agents/${agentId}/speakers/${speakerId}`, {
        method: "PUT",
        body: JSON.stringify(input),
      }),
    onSuccess: (speaker) => {
      queryClient.setQueryData<Speaker[]>(speakersQuery(agentId).queryKey, (old) =>
        old?.map((s) => (s.id === speaker.id ? speaker : s)),
      );
    },
  });
}

export function useDeleteSpeaker(agentId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (speakerId: number) =>
      api<void>(`/api/agents/${agentId}/speakers/${speakerId}`, { method: "DELETE" }),
    onSuccess: (_data, speakerId) => {
      queryClient.setQueryData<Speaker[]>(speakersQuery(agentId).queryKey, (old) =>
        old?.filter((s) => s.id !== speakerId),
      );
    },
  });
}

export function describeSpeakerError(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  return copy.common.network;
}

export function describeEnrollError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 400) {
      const body = err.message.replace(/^Bad request: /, "");
      const prefix = "rejected message ids: ";
      if (body.startsWith(prefix)) {
        const ids = body
          .slice(prefix.length)
          .split(", ")
          .map((part) => /^(\d+): /.exec(part)?.[1])
          .filter((id): id is string => id != null)
          .map(Number);
        if (ids.length > 0) return copy.agents.speakers.errors.rejected([...new Set(ids)]);
      }
      if (body === "audio capture is not enabled") {
        return copy.agents.speakers.errors.captureDisabled;
      }
      return err.message;
    }
    if (err.status === 503) return copy.agents.speakers.errors.modelMissing;
    return err.message;
  }
  return copy.common.network;
}

export function describeUpdateSpeakerError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 404) return copy.agents.speakers.errors.notFound;
    if (err.status === 409) return copy.agents.speakers.errors.nameTaken;
    return err.message;
  }
  return describeSpeakerError(err);
}

export function describeDeleteSpeakerError(err: unknown): string {
  if (err instanceof ApiError && err.status === 404) {
    return copy.agents.speakers.errors.notFound;
  }
  return describeSpeakerError(err);
}
