import {
  useInfiniteQuery,
  useMutation,
  useQueryClient,
  type InfiniteData,
} from "@tanstack/react-query";
import { ApiError, api } from "./api";
import { copy } from "./copy";

export interface Conversation {
  id: number;
  session_id: string;
  device_id: number;
  created_at: string;
  ended_at: string | null;
}

export interface ConversationPage {
  items: Conversation[];
  has_more: boolean;
}

export interface ConversationMessage {
  id: number;
  role: string;
  message: unknown;
  speaker_id: number | null;
  has_audio: boolean;
  created_at: string;
}

export interface MessagePage {
  items: ConversationMessage[];
  has_more: boolean;
}

export interface ToolCallPayload {
  id: string;
  name: string;
  arguments: string;
}

export function messageText(message: ConversationMessage): string | null {
  const payload = message.message as { content?: unknown } | null;
  const content = payload?.content;
  return typeof content === "string" ? content : null;
}

export function messageToolCalls(message: ConversationMessage): ToolCallPayload[] {
  const payload = message.message as { tool_calls?: unknown } | null;
  return Array.isArray(payload?.tool_calls) ? (payload.tool_calls as ToolCallPayload[]) : [];
}

function conversationsKey(agentId: number) {
  return ["agents", agentId, "conversations"] as const;
}

export function useConversations(agentId: number) {
  return useInfiniteQuery({
    queryKey: conversationsKey(agentId),
    queryFn: ({ pageParam }) =>
      api<ConversationPage>(
        `/api/agents/${agentId}/conversations${pageParam == null ? "" : `?before_id=${pageParam}`}`,
      ),
    initialPageParam: undefined as number | undefined,
    getNextPageParam: (last) => (last.has_more ? (last.items.at(-1)?.id ?? undefined) : undefined),
  });
}

export function useMessages(agentId: number, conversationId: number | null) {
  return useInfiniteQuery({
    queryKey: ["agents", agentId, "conversations", conversationId, "messages"] as const,
    queryFn: ({ pageParam }) =>
      api<MessagePage>(
        `/api/agents/${agentId}/conversations/${conversationId}/messages${
          pageParam == null ? "" : `?before_id=${pageParam}`
        }`,
      ),
    initialPageParam: undefined as number | undefined,
    getNextPageParam: (last) => (last.has_more ? (last.items.at(-1)?.id ?? undefined) : undefined),
    enabled: conversationId != null,
  });
}

export function useDeleteConversation(agentId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (conversationId: number) =>
      api<void>(`/api/agents/${agentId}/conversations/${conversationId}`, { method: "DELETE" }),
    onSuccess: (_data, conversationId) => {
      queryClient.setQueryData<InfiniteData<ConversationPage>>(conversationsKey(agentId), (old) =>
        old
          ? {
              ...old,
              pages: old.pages.map((page) => ({
                ...page,
                items: page.items.filter((c) => c.id !== conversationId),
              })),
            }
          : old,
      );
      queryClient.removeQueries({
        queryKey: ["agents", agentId, "conversations", conversationId],
      });
    },
  });
}

export async function fetchMessageAudio(
  agentId: number,
  conversationId: number,
  messageId: number,
): Promise<Blob> {
  const res = await fetch(
    `/api/agents/${agentId}/conversations/${conversationId}/messages/${messageId}/audio`,
  );
  if (!res.ok) {
    const body = (await res.json().catch(() => null)) as { error?: string } | null;
    throw new ApiError(
      res.status,
      body?.error ?? copy.common.requestFailed(res.status),
      Number(res.headers.get("retry-after")) || undefined,
    );
  }
  return res.blob();
}

export function describeConversationError(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  return copy.common.network;
}

export function describeDeleteConversationError(err: unknown): string {
  if (err instanceof ApiError && err.status === 404) {
    return copy.agents.conversations.errors.notFound;
  }
  return describeConversationError(err);
}

export function describeMessageError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 404) return copy.agents.conversations.errors.notFound;
    return err.message;
  }
  return copy.common.network;
}

export function describeAudioError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 404) return copy.agents.conversations.errors.audioMissing;
    if (err.status === 503) return copy.agents.conversations.errors.audioUnavailable;
    return err.message;
  }
  return copy.common.network;
}
