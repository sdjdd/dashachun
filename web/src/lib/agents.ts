import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { z } from "zod";
import { ApiError, api } from "./api";
import { copy } from "./copy";

export interface Agent {
  id: number;
  name: string;
  persona_prompt: string;
  created_at: string;
}

export const agentSchema = z.object({
  name: z
    .string()
    .min(1, copy.agents.validation.nameRequired)
    .max(64, copy.agents.validation.nameLength),
  persona_prompt: z.string().max(4096, copy.agents.validation.personaLength),
});

export type AgentInput = z.infer<typeof agentSchema>;

async function fetchAgents(): Promise<Agent[]> {
  return api<Agent[]>("/api/agents");
}

export const agentsQuery = {
  queryKey: ["agents"] as const,
  queryFn: fetchAgents,
};

export function useAgents() {
  return useQuery(agentsQuery);
}

export function agentQuery(id: number) {
  return {
    queryKey: ["agents", id] as const,
    queryFn: () => api<Agent>(`/api/agents/${id}`),
  };
}

export function useAgent(id: number) {
  const valid = Number.isSafeInteger(id) && id > 0;
  return useQuery({ ...agentQuery(id), enabled: valid });
}

export function useCreateAgent() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (input: AgentInput) =>
      api<Agent>("/api/agents", { method: "POST", body: JSON.stringify(input) }),
    onSuccess: (agent) => {
      queryClient.setQueryData<Agent[]>(agentsQuery.queryKey, (old) =>
        old ? [...old, agent] : [agent],
      );
    },
  });
}

export function useUpdateAgent() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, input }: { id: number; input: AgentInput }) =>
      api<Agent>(`/api/agents/${id}`, { method: "PUT", body: JSON.stringify(input) }),
    onSuccess: (agent) => {
      queryClient.setQueryData<Agent[]>(agentsQuery.queryKey, (old) =>
        old?.map((a) => (a.id === agent.id ? agent : a)),
      );
      queryClient.setQueryData(agentQuery(agent.id).queryKey, agent);
    },
  });
}

export function useDeleteAgent() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api<void>(`/api/agents/${id}`, { method: "DELETE" }),
    onSuccess: (_data, id) => {
      queryClient.setQueryData<Agent[]>(agentsQuery.queryKey, (old) =>
        old?.filter((a) => a.id !== id),
      );
    },
  });
}

export function describeAgentError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 409) return copy.agents.errors.nameTaken;
    return err.message;
  }
  return copy.common.network;
}

export function isAgentNotFound(err: unknown): boolean {
  return err instanceof ApiError && err.status === 404;
}
