import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ApiError, api } from "./api";
import { copy } from "./copy";

export interface MemoryEntry {
  mem_no: number;
  content: string;
}

function memoriesQuery(agentId: number) {
  return {
    queryKey: ["agents", agentId, "memories"] as const,
    queryFn: () => api<MemoryEntry[]>(`/api/agents/${agentId}/memories`),
  };
}

export function useMemories(agentId: number) {
  return useQuery(memoriesQuery(agentId));
}

export function useDeleteMemory(agentId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (memNo: number) =>
      api<void>(`/api/agents/${agentId}/memories/${memNo}`, { method: "DELETE" }),
    onSuccess: (_data, memNo) => {
      queryClient.setQueryData<MemoryEntry[]>(memoriesQuery(agentId).queryKey, (old) =>
        old?.filter((m) => m.mem_no !== memNo),
      );
    },
  });
}

export function describeMemoryError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 404) return copy.agents.memories.errors.notFound;
    return err.message;
  }
  return copy.common.network;
}
