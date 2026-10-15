import { useQuery } from "@tanstack/react-query";
import { api } from "./api";

export interface Speaker {
  id: number;
  agent_id: number;
  name: string;
  description: string;
  sample_count: number;
  created_at: string;
  updated_at: string;
}

export function useSpeakers(agentId: number) {
  return useQuery({
    queryKey: ["agents", agentId, "speakers"] as const,
    queryFn: () => api<Speaker[]>(`/api/agents/${agentId}/speakers`),
  });
}
