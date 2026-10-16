import { createFileRoute } from "@tanstack/react-router";
import { Controller, useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { AudioWaveform, Check, LoaderCircle, Pencil, Trash2 } from "lucide-react";
import { useState } from "react";
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
import { EmptyState, ErrorState, LoadingState } from "@/components/states";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  describeMessageError,
  messageText,
  useConversations,
  useMessages,
} from "@/lib/conversations";
import type { Conversation, ConversationMessage } from "@/lib/conversations";
import { copy } from "@/lib/copy";
import {
  createSpeakerSchema,
  describeDeleteSpeakerError,
  describeEnrollError,
  describeSpeakerError,
  describeUpdateSpeakerError,
  updateSpeakerSchema,
  useDeleteSpeaker,
  useEnrollSpeaker,
  useSpeakers,
  useUpdateSpeaker,
} from "@/lib/speakers";
import type { CreateSpeakerInput, Speaker, UpdateSpeakerInput } from "@/lib/speakers";

export const Route = createFileRoute("/_app/agents/$agentId/speakers")({
  component: SpeakersTab,
});

function formatDateTime(iso: string): string {
  return new Date(iso).toLocaleString("zh-CN", {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function SpeakersTab() {
  const { agentId } = Route.useParams();
  const id = Number(agentId);
  const speakers = useSpeakers(id);

  if (speakers.isPending) {
    return <LoadingState />;
  }
  if (speakers.isError) {
    return (
      <ErrorState
        message={describeSpeakerError(speakers.error)}
        onRetry={() => speakers.refetch()}
      />
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <AudioWaveform className="size-4 text-muted-foreground" />
          {copy.agents.speakers.listTitle}
        </CardTitle>
        <CardDescription>{copy.agents.speakers.listDescription}</CardDescription>
        <CardAction>
          <EnrollDialog agentId={id} />
        </CardAction>
      </CardHeader>
      <CardContent>
        {speakers.data.length === 0 ? (
          <EmptyState
            icon={AudioWaveform}
            text={copy.agents.speakers.empty}
            hint={copy.agents.speakers.emptyHint}
          />
        ) : (
          <ul className="divide-y">
            {speakers.data.map((speaker) => (
              <SpeakerRow key={speaker.id} agentId={id} speaker={speaker} />
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

function SpeakerRow({ agentId, speaker }: { agentId: number; speaker: Speaker }) {
  const remove = useDeleteSpeaker(agentId);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  return (
    <li className="flex flex-wrap items-center justify-between gap-2 py-3 first:pt-0 last:pb-0">
      <div className="flex min-w-0 items-center gap-3">
        <div className="grid size-9 shrink-0 place-items-center rounded-full bg-primary/10 text-sm font-medium text-primary">
          {speaker.name.slice(0, 1)}
        </div>
        <div className="grid min-w-0 gap-0.5">
          <span className="truncate text-sm font-medium">{speaker.name}</span>
          {speaker.description && (
            <span className="truncate text-sm text-muted-foreground">{speaker.description}</span>
          )}
          <span className="text-xs text-muted-foreground">
            {copy.agents.speakers.samples(speaker.sample_count)} ·{" "}
            {new Date(speaker.created_at).toLocaleDateString("zh-CN")}
          </span>
        </div>
      </div>
      <div className="flex items-center gap-1">
        <EditSpeakerDialog agentId={agentId} speaker={speaker} />
        <Button
          variant="ghost"
          size="icon-sm"
          className="text-muted-foreground hover:text-destructive"
          aria-label={copy.agents.speakers.deleteTitle}
          onClick={() => {
            setError(null);
            setConfirmOpen(true);
          }}
        >
          <Trash2 className="size-4" />
        </Button>
      </div>

      <AlertDialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{copy.agents.speakers.deleteTitle}</AlertDialogTitle>
            <AlertDialogDescription>
              {copy.agents.speakers.deleteDescription(speaker.name)}
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
                  await remove.mutateAsync(speaker.id);
                  setConfirmOpen(false);
                } catch (err) {
                  setError(describeDeleteSpeakerError(err));
                }
              }}
            >
              {remove.isPending && <LoaderCircle className="animate-spin" />}
              {copy.agents.speakers.deleteConfirm}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </li>
  );
}

function EnrollDialog({ agentId }: { agentId: number }) {
  const enroll = useEnrollSpeaker(agentId);
  const [open, setOpen] = useState(false);
  const form = useForm<CreateSpeakerInput>({
    resolver: zodResolver(createSpeakerSchema),
    defaultValues: { name: "", description: "", message_ids: [] },
  });
  const errors = form.formState.errors;

  async function onSubmit(values: CreateSpeakerInput) {
    form.clearErrors("root");
    try {
      await enroll.mutateAsync(values);
      setOpen(false);
    } catch (err) {
      form.setError("root", { message: describeEnrollError(err) });
    }
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        setOpen(next);
        if (next) {
          form.reset({ name: "", description: "", message_ids: [] });
        }
      }}
    >
      <DialogTrigger render={<Button size="sm" />}>{copy.agents.speakers.enroll}</DialogTrigger>
      <DialogContent className="sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle>{copy.agents.speakers.enroll}</DialogTitle>
          <DialogDescription>{copy.agents.speakers.enrollDescription}</DialogDescription>
        </DialogHeader>
        <form onSubmit={form.handleSubmit(onSubmit)} className="grid gap-4" noValidate>
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="grid gap-2">
              <Label htmlFor="speaker-name">{copy.agents.speakers.nameLabel}</Label>
              <Input
                id="speaker-name"
                placeholder={copy.agents.speakers.namePlaceholder}
                {...form.register("name")}
                aria-invalid={!!errors.name}
              />
              {errors.name && <p className="text-sm text-destructive">{errors.name.message}</p>}
            </div>
            <div className="grid gap-2">
              <div className="flex items-baseline justify-between">
                <Label htmlFor="speaker-description">{copy.agents.speakers.descriptionLabel}</Label>
                <span className="text-xs text-muted-foreground">
                  {copy.agents.speakers.descriptionOptional}
                </span>
              </div>
              <Input
                id="speaker-description"
                placeholder={copy.agents.speakers.descriptionPlaceholder}
                {...form.register("description")}
                aria-invalid={!!errors.description}
              />
              {errors.description && (
                <p className="text-sm text-destructive">{errors.description.message}</p>
              )}
            </div>
          </div>
          <Controller
            control={form.control}
            name="message_ids"
            render={({ field }) => (
              <AudioPicker agentId={agentId} value={field.value} onChange={field.onChange} />
            )}
          />
          {errors.message_ids && (
            <p className="text-sm text-destructive">{errors.message_ids.message}</p>
          )}
          {errors.root && <p className="text-sm text-destructive">{errors.root.message}</p>}
          <DialogFooter>
            <DialogClose render={<Button variant="outline" type="button" />}>
              {copy.agents.cancel}
            </DialogClose>
            <Button type="submit" disabled={form.formState.isSubmitting}>
              {form.formState.isSubmitting && <LoaderCircle className="animate-spin" />}
              {copy.agents.speakers.enrollSubmit}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

function AudioPicker({
  agentId,
  value,
  onChange,
}: {
  agentId: number;
  value: number[];
  onChange: (ids: number[]) => void;
}) {
  const conversations = useConversations(agentId);
  const items = conversations.data?.pages.flatMap((page) => page.items) ?? [];
  const [picked, setPicked] = useState<number | null>(null);
  const selected = picked ?? items[0]?.id ?? null;
  const messages = useMessages(agentId, selected);
  const pickable =
    messages.data?.pages
      .flatMap((page) => page.items)
      .filter((message) => message.role === "user" && message.has_audio) ?? [];

  const toggle = (messageId: number) => {
    if (value.includes(messageId)) {
      onChange(value.filter((id) => id !== messageId));
    } else if (value.length < 10) {
      onChange([...value, messageId]);
    }
  };

  return (
    <div className="grid gap-2">
      <div className="flex items-baseline justify-between">
        <span className="text-sm font-medium">{copy.agents.speakers.pickerTitle}</span>
        <span className="text-xs text-muted-foreground">
          {copy.agents.speakers.selectedCount(value.length)}
        </span>
      </div>
      <div className="grid gap-2 sm:grid-cols-[220px_minmax(0,1fr)]">
        <div className="grid content-start gap-1">
          <span className="text-xs text-muted-foreground">
            {copy.agents.speakers.pickerConversations}
          </span>
          <div className="max-h-60 overflow-y-auto rounded-lg border">
            {conversations.isPending ? (
              <div className="grid place-items-center py-8 text-muted-foreground">
                <LoaderCircle className="animate-spin" />
              </div>
            ) : conversations.isError ? (
              <p className="p-3 text-sm text-destructive">
                {describeMessageError(conversations.error)}
              </p>
            ) : items.length === 0 ? (
              <p className="p-3 text-sm text-muted-foreground">
                {copy.agents.speakers.pickerEmpty}
              </p>
            ) : (
              <ul className="divide-y">
                {items.map((conversation) => (
                  <ConversationPickerRow
                    key={conversation.id}
                    conversation={conversation}
                    selected={conversation.id === selected}
                    onSelect={() => setPicked(conversation.id)}
                  />
                ))}
              </ul>
            )}
            {conversations.hasNextPage && (
              <div className="border-t p-1 text-center">
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={conversations.isFetchingNextPage}
                  onClick={() => conversations.fetchNextPage()}
                >
                  {conversations.isFetchingNextPage && <LoaderCircle className="animate-spin" />}
                  {copy.agents.conversations.loadEarlierConversations}
                </Button>
              </div>
            )}
          </div>
        </div>
        <div className="grid content-start gap-1">
          <span className="text-xs text-muted-foreground">
            {copy.agents.speakers.pickerMessages}
          </span>
          <div className="max-h-60 overflow-y-auto rounded-lg border">
            {selected == null ? (
              <p className="p-3 text-sm text-muted-foreground">
                {copy.agents.speakers.pickerSelectConversation}
              </p>
            ) : messages.isPending ? (
              <div className="grid place-items-center py-8 text-muted-foreground">
                <LoaderCircle className="animate-spin" />
              </div>
            ) : messages.isError ? (
              <p className="p-3 text-sm text-destructive">{describeMessageError(messages.error)}</p>
            ) : pickable.length === 0 ? (
              <p className="p-3 text-sm text-muted-foreground">
                {copy.agents.speakers.pickerNoMessages}
              </p>
            ) : (
              <ul className="divide-y">
                {pickable.map((message) => (
                  <MessagePickerRow
                    key={message.id}
                    message={message}
                    selected={value.includes(message.id)}
                    atLimit={value.length >= 10}
                    onToggle={() => toggle(message.id)}
                  />
                ))}
              </ul>
            )}
            {selected != null && messages.hasNextPage && (
              <div className="border-t p-1 text-center">
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={messages.isFetchingNextPage}
                  onClick={() => messages.fetchNextPage()}
                >
                  {messages.isFetchingNextPage && <LoaderCircle className="animate-spin" />}
                  {copy.agents.conversations.loadEarlierMessages}
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function ConversationPickerRow({
  conversation,
  selected,
  onSelect,
}: {
  conversation: Conversation;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <li>
      <button
        type="button"
        onClick={onSelect}
        data-selected={selected}
        className="w-full px-3 py-2 text-left transition-colors hover:bg-muted/50 data-[selected=true]:bg-muted"
      >
        <span className="text-sm font-medium">#{conversation.id}</span>
        <span className="block text-xs text-muted-foreground">
          {formatDateTime(conversation.created_at)}
        </span>
      </button>
    </li>
  );
}

function MessagePickerRow({
  message,
  selected,
  atLimit,
  onToggle,
}: {
  message: ConversationMessage;
  selected: boolean;
  atLimit: boolean;
  onToggle: () => void;
}) {
  const text = messageText(message);

  return (
    <li>
      <button
        type="button"
        onClick={onToggle}
        disabled={!selected && atLimit}
        data-selected={selected}
        className="w-full px-3 py-2 text-left transition-colors hover:bg-muted/50 disabled:cursor-not-allowed disabled:opacity-50 data-[selected=true]:bg-muted"
      >
        <div className="flex items-start justify-between gap-2">
          <div className="grid gap-0.5">
            <span className="text-xs text-muted-foreground">
              {formatDateTime(message.created_at)}
            </span>
            <span className="line-clamp-2 text-sm">{text ?? "—"}</span>
          </div>
          {selected && <Check className="mt-0.5 size-4 shrink-0" />}
        </div>
      </button>
    </li>
  );
}

function EditSpeakerDialog({ agentId, speaker }: { agentId: number; speaker: Speaker }) {
  const update = useUpdateSpeaker(agentId);
  const [open, setOpen] = useState(false);
  const form = useForm<UpdateSpeakerInput>({
    resolver: zodResolver(updateSpeakerSchema),
    defaultValues: { name: speaker.name, description: speaker.description },
  });
  const errors = form.formState.errors;

  async function onSubmit(values: UpdateSpeakerInput) {
    form.clearErrors("root");
    try {
      await update.mutateAsync({ speakerId: speaker.id, input: values });
      setOpen(false);
    } catch (err) {
      form.setError("root", { message: describeUpdateSpeakerError(err) });
    }
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        setOpen(next);
        if (next) {
          form.reset({ name: speaker.name, description: speaker.description });
        }
      }}
    >
      <DialogTrigger
        render={<Button variant="ghost" size="icon-sm" aria-label={copy.agents.speakers.edit} />}
      >
        <Pencil className="size-4 text-muted-foreground" />
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{copy.agents.speakers.editTitle}</DialogTitle>
        </DialogHeader>
        <form onSubmit={form.handleSubmit(onSubmit)} className="grid gap-4" noValidate>
          <div className="grid gap-2">
            <Label htmlFor="speaker-edit-name">{copy.agents.speakers.nameLabel}</Label>
            <Input
              id="speaker-edit-name"
              placeholder={copy.agents.speakers.namePlaceholder}
              {...form.register("name")}
              aria-invalid={!!errors.name}
            />
            {errors.name && <p className="text-sm text-destructive">{errors.name.message}</p>}
          </div>
          <div className="grid gap-2">
            <div className="flex items-baseline justify-between">
              <Label htmlFor="speaker-edit-description">
                {copy.agents.speakers.descriptionLabel}
              </Label>
              <span className="text-xs text-muted-foreground">
                {copy.agents.speakers.descriptionOptional}
              </span>
            </div>
            <Input
              id="speaker-edit-description"
              placeholder={copy.agents.speakers.descriptionPlaceholder}
              {...form.register("description")}
              aria-invalid={!!errors.description}
            />
            {errors.description && (
              <p className="text-sm text-destructive">{errors.description.message}</p>
            )}
          </div>
          {errors.root && <p className="text-sm text-destructive">{errors.root.message}</p>}
          <DialogFooter>
            <DialogClose render={<Button variant="outline" type="button" />}>
              {copy.agents.cancel}
            </DialogClose>
            <Button type="submit" disabled={form.formState.isSubmitting}>
              {form.formState.isSubmitting && <LoaderCircle className="animate-spin" />}
              {copy.agents.speakers.editSubmit}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
