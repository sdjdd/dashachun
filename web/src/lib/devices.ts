import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { z } from "zod";
import { ApiError, api } from "./api";
import { copy } from "./copy";

export interface Device {
  client_id: string;
  device_id: string | null;
  board_type: string | null;
  agent_id: number | null;
  activated_at: string | null;
  created_at: string;
}

export const deviceCodeSchema = z.object({
  code: z.string().regex(/^[0-9]{6}$/, copy.agents.devices.validation.code),
});

export type DeviceCodeInput = z.infer<typeof deviceCodeSchema>;

function devicesQuery(agentId: number) {
  return {
    queryKey: ["agents", agentId, "devices"] as const,
    queryFn: () => api<Device[]>(`/api/agents/${agentId}/devices`),
  };
}

export function useDevices(agentId: number) {
  return useQuery(devicesQuery(agentId));
}

export function useBindDevice(agentId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (input: DeviceCodeInput) =>
      api<Device>(`/api/agents/${agentId}/devices`, {
        method: "POST",
        body: JSON.stringify(input),
      }),
    onSuccess: (device) => {
      queryClient.setQueryData<Device[]>(devicesQuery(agentId).queryKey, (old) =>
        old?.some((d) => d.client_id === device.client_id) ? old : [...(old ?? []), device],
      );
    },
  });
}

export function useUnbindDevice(agentId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (clientId: string) =>
      api<void>(`/api/agents/${agentId}/devices/${clientId}`, { method: "DELETE" }),
    onSuccess: (_data, clientId) => {
      queryClient.setQueryData<Device[]>(devicesQuery(agentId).queryKey, (old) =>
        old?.filter((d) => d.client_id !== clientId),
      );
    },
  });
}

export function describeDeviceError(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  return copy.common.network;
}

export function describeDeviceBindError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 404) return copy.agents.devices.errors.codeInvalid;
    if (err.status === 409) return copy.agents.devices.errors.alreadyBound;
  }
  return describeDeviceError(err);
}

export function describeUnbindError(err: unknown): string {
  if (err instanceof ApiError && err.status === 404) {
    return copy.agents.devices.errors.notBoundHere;
  }
  return describeDeviceError(err);
}
