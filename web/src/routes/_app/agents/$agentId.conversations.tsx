import { createFileRoute } from "@tanstack/react-router";
import { LoaderCircle, Play, Square, Trash2, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { z } from "zod";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  describeAudioError,
  describeConversationError,
  describeDeleteConversationError,
  describeMessageError,
  fetchMessageAudio,
  messageText,
  messageToolCalls,
  useConversations,
  useDeleteConversation,
  useMessages,
} from "@/lib/conversations";
import type { Conversation, ConversationMessage } from "@/lib/conversations";
import { copy } from "@/lib/copy";
import { useSpeakers } from "@/lib/speakers";
import type { Speaker } from "@/lib/speakers";

const conversationsSearchSchema = z.object({
  c: z.coerce.number().int().positive().optional(),
});

export const Route = createFileRoute("/_app/agents/$agentId/conversations")({
  validateSearch: (search) => {
    const parsed = conversationsSearchSchema.safeParse(search);
    return parsed.success ? parsed.data : {};
  },
  component: ConversationsTab,
});

function formatDateTime(iso: string): string {
  return new Date(iso).toLocaleString("zh-CN", {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function ConversationsTab() {
  const { agentId } = Route.useParams();
  const id = Number(agentId);
  const { c } = Route.useSearch();
  const selected = typeof c === "number" ? c : null;
  const navigate = Route.useNavigate();
  const conversations = useConversations(id);
  const speakers = useSpeakers(id);

  const select = (conversationId: number | null) => {
    navigate({
      search: (prev) => ({ ...prev, c: conversationId ?? undefined }),
    });
  };

  if (conversations.isPending) {
    return (
      <div className="grid place-items-center py-16 text-muted-foreground">
        <LoaderCircle className="animate-spin" />
      </div>
    );
  }
  if (conversations.isError) {
    return (
      <div className="grid place-items-center gap-3 py-16">
        <p className="text-sm text-destructive">{describeConversationError(conversations.error)}</p>
        <Button variant="outline" size="sm" onClick={() => conversations.refetch()}>
          {copy.agents.retry}
        </Button>
      </div>
    );
  }

  const items = conversations.data.pages.flatMap((page) => page.items);

  return (
    <div className="grid gap-4 lg:grid-cols-[340px_minmax(0,1fr)] lg:items-start">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">{copy.agents.conversations.listTitle}</CardTitle>
          <CardDescription>{copy.agents.conversations.listDescription}</CardDescription>
        </CardHeader>
        <CardContent>
          {items.length === 0 ? (
            <div className="grid place-items-center gap-1 py-8 text-center">
              <p className="text-sm text-muted-foreground">{copy.agents.conversations.empty}</p>
              <p className="text-xs text-muted-foreground">{copy.agents.conversations.emptyHint}</p>
            </div>
          ) : (
            <div className="grid gap-1">
              <ul className="divide-y">
                {items.map((conversation) => (
                  <ConversationRow
                    key={conversation.id}
                    agentId={id}
                    conversation={conversation}
                    selected={conversation.id === selected}
                    onSelect={() => select(conversation.id)}
                    onDeleted={() => {
                      if (conversation.id === selected) select(null);
                    }}
                  />
                ))}
              </ul>
              {conversations.hasNextPage && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="mt-2"
                  disabled={conversations.isFetchingNextPage}
                  onClick={() => conversations.fetchNextPage()}
                >
                  {conversations.isFetchingNextPage && <LoaderCircle className="animate-spin" />}
                  {copy.agents.conversations.loadEarlierConversations}
                </Button>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {selected == null ? (
        <div className="grid place-items-center rounded-xl border border-dashed py-16 text-sm text-muted-foreground">
          {copy.agents.conversations.selectHint}
        </div>
      ) : (
        <TranscriptPane
          agentId={id}
          conversationId={selected}
          speakers={speakers.data ?? []}
          onClose={() => select(null)}
        />
      )}
    </div>
  );
}

function ConversationRow({
  agentId,
  conversation,
  selected,
  onSelect,
  onDeleted,
}: {
  agentId: number;
  conversation: Conversation;
  selected: boolean;
  onSelect: () => void;
  onDeleted: () => void;
}) {
  const remove = useDeleteConversation(agentId);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  return (
    <li className="relative">
      <button
        type="button"
        onClick={onSelect}
        data-selected={selected}
        className="w-full rounded-md py-3 pr-9 text-left transition-colors hover:bg-muted/50 data-[selected=true]:bg-muted"
      >
        <div className="grid gap-0.5">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium">#{conversation.id}</span>
            <span
              className={
                conversation.ended_at
                  ? "text-xs text-muted-foreground"
                  : "text-xs font-medium text-primary"
              }
            >
              {conversation.ended_at
                ? copy.agents.conversations.ended
                : copy.agents.conversations.ongoing}
            </span>
          </div>
          <span className="text-xs text-muted-foreground">
            {formatDateTime(conversation.created_at)} · 设备 #{conversation.device_id}
          </span>
        </div>
      </button>
      <Button
        variant="ghost"
        size="icon-sm"
        className="absolute top-1/2 right-1 -translate-y-1/2"
        aria-label={copy.agents.conversations.deleteTitle}
        onClick={() => {
          setError(null);
          setConfirmOpen(true);
        }}
      >
        <Trash2 className="size-4" />
      </Button>

      <AlertDialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{copy.agents.conversations.deleteTitle}</AlertDialogTitle>
            <AlertDialogDescription>
              {copy.agents.conversations.deleteDescription}
            </AlertDialogDescription>
          </AlertDialogHeader>
          {error && <p className="text-sm text-destructive">{error}</p>}
          <AlertDialogFooter>
            <AlertDialogCancel>{copy.agents.cancel}</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              disabled={remove.isPending}
              onClick={async () => {
                setError(null);
                try {
                  await remove.mutateAsync(conversation.id);
                  setConfirmOpen(false);
                  onDeleted();
                } catch (err) {
                  setError(describeDeleteConversationError(err));
                }
              }}
            >
              {remove.isPending && <LoaderCircle className="animate-spin" />}
              {copy.agents.conversations.deleteConfirm}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </li>
  );
}

function TranscriptPane({
  agentId,
  conversationId,
  speakers,
  onClose,
}: {
  agentId: number;
  conversationId: number;
  speakers: Speaker[];
  onClose: () => void;
}) {
  const messages = useMessages(agentId, conversationId);
  const bottomRef = useRef<HTMLDivElement>(null);
  const scrolledFor = useRef<number | null>(null);

  useEffect(() => {
    if (messages.isSuccess && scrolledFor.current !== conversationId) {
      scrolledFor.current = conversationId;
      bottomRef.current?.scrollIntoView({ block: "end" });
    }
  }, [messages.isSuccess, conversationId]);

  if (messages.isPending) {
    return (
      <div className="grid place-items-center py-16 text-muted-foreground">
        <LoaderCircle className="animate-spin" />
      </div>
    );
  }
  if (messages.isError) {
    return (
      <div className="grid place-items-center gap-3 py-16">
        <p className="text-sm text-destructive">{describeMessageError(messages.error)}</p>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => messages.refetch()}>
            {copy.agents.retry}
          </Button>
          <Button variant="ghost" size="sm" onClick={onClose}>
            <X className="size-4" />
          </Button>
        </div>
      </div>
    );
  }

  const chronological = messages.data
    ? [...messages.data.pages].reverse().flatMap((page) => [...page.items].reverse())
    : [];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">
          {copy.agents.conversations.transcriptTitle} · #{conversationId}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="max-h-[70vh] overflow-y-auto">
          {messages.hasNextPage && (
            <div className="pb-2 text-center">
              <Button
                variant="outline"
                size="sm"
                disabled={messages.isFetchingNextPage}
                onClick={() => messages.fetchNextPage()}
              >
                {messages.isFetchingNextPage && <LoaderCircle className="animate-spin" />}
                {copy.agents.conversations.loadEarlierMessages}
              </Button>
            </div>
          )}
          <ul className="divide-y">
            {chronological.map((message) => (
              <MessageRow
                key={message.id}
                agentId={agentId}
                conversationId={conversationId}
                message={message}
                speaker={speakers.find((s) => s.id === message.speaker_id) ?? null}
              />
            ))}
          </ul>
          <div ref={bottomRef} />
        </div>
      </CardContent>
    </Card>
  );
}

function roleLabel(role: string): string {
  const roles = copy.agents.conversations.roles;
  if (role === "user" || role === "assistant" || role === "tool") return roles[role];
  return role;
}

function MessageRow({
  agentId,
  conversationId,
  message,
  speaker,
}: {
  agentId: number;
  conversationId: number;
  message: ConversationMessage;
  speaker: Speaker | null;
}) {
  const text = messageText(message);
  const toolCalls = message.role === "assistant" ? messageToolCalls(message) : [];
  const isTool = message.role === "tool";

  return (
    <li className="grid gap-1.5 py-3 first:pt-0 last:pb-0">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-medium text-muted-foreground">{roleLabel(message.role)}</span>
        {message.role === "user" && message.speaker_id != null && (
          <span className="text-xs text-muted-foreground">
            {speaker?.name ?? copy.agents.conversations.unknownSpeaker}
          </span>
        )}
        <span className="text-xs text-muted-foreground">{formatDateTime(message.created_at)}</span>
      </div>
      {isTool ? (
        <p className="rounded-md bg-muted/50 px-2 py-1.5 font-mono text-xs break-all whitespace-pre-wrap text-muted-foreground">
          {text ?? ""}
        </p>
      ) : (
        text != null && <p className="text-sm whitespace-pre-wrap">{text}</p>
      )}
      {toolCalls.map((call) => (
        <p key={call.id} className="text-xs text-muted-foreground">
          {copy.agents.conversations.toolCall(call.name)}
        </p>
      ))}
      {message.has_audio && (
        <AudioControls agentId={agentId} conversationId={conversationId} messageId={message.id} />
      )}
    </li>
  );
}

function AudioControls({
  agentId,
  conversationId,
  messageId,
}: {
  agentId: number;
  conversationId: number;
  messageId: number;
}) {
  const [audio, setAudio] = useState<{ url: string; el: HTMLAudioElement } | null>(null);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(
    () => () => {
      audio?.el.pause();
      if (audio) URL.revokeObjectURL(audio.url);
    },
    [audio],
  );

  return (
    <div className="grid gap-1.5 justify-items-start">
      <Button
        variant="outline"
        size="sm"
        disabled={loading}
        onClick={async () => {
          setError(null);
          if (audio && playing) {
            audio.el.pause();
            audio.el.currentTime = 0;
            setPlaying(false);
            return;
          }
          if (audio) {
            await audio.el.play();
            setPlaying(true);
            return;
          }
          setLoading(true);
          try {
            const blob = await fetchMessageAudio(agentId, conversationId, messageId);
            const url = URL.createObjectURL(blob);
            const el = new Audio(url);
            el.onended = () => setPlaying(false);
            setAudio({ url, el });
            setPlaying(true);
            el.play().catch(() => {
              URL.revokeObjectURL(url);
              setAudio(null);
              setPlaying(false);
            });
          } catch (err) {
            setError(describeAudioError(err));
          } finally {
            setLoading(false);
          }
        }}
      >
        {loading && <LoaderCircle className="animate-spin" />}
        {!loading && (playing ? <Square className="size-4" /> : <Play className="size-4" />)}
        {playing ? copy.agents.conversations.stopAudio : copy.agents.conversations.playAudio}
      </Button>
      {error && <p className="text-sm text-destructive">{error}</p>}
    </div>
  );
}
