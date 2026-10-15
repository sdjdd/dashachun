import { createFileRoute } from "@tanstack/react-router";
import { LoaderCircle, Trash2 } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
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
import { describeMemoryError, useDeleteMemory, useMemories } from "@/lib/memories";
import type { MemoryEntry } from "@/lib/memories";
import { copy } from "@/lib/copy";

export const Route = createFileRoute("/_app/agents/$agentId/memories")({
  component: MemoriesTab,
});

function MemoriesTab() {
  const { agentId } = Route.useParams();
  const id = Number(agentId);
  const memories = useMemories(id);

  if (memories.isPending) {
    return (
      <div className="grid place-items-center py-16 text-muted-foreground">
        <LoaderCircle className="animate-spin" />
      </div>
    );
  }
  if (memories.isError) {
    return (
      <div className="grid place-items-center gap-3 py-16">
        <p className="text-sm text-destructive">{describeMemoryError(memories.error)}</p>
        <Button variant="outline" size="sm" onClick={() => memories.refetch()}>
          {copy.agents.retry}
        </Button>
      </div>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{copy.agents.memories.title}</CardTitle>
        <CardDescription>{copy.agents.memories.description}</CardDescription>
      </CardHeader>
      <CardContent>
        {memories.data.length === 0 ? (
          <div className="grid place-items-center gap-1 py-8 text-center">
            <p className="text-sm text-muted-foreground">{copy.agents.memories.empty}</p>
            <p className="text-xs text-muted-foreground">{copy.agents.memories.emptyHint}</p>
          </div>
        ) : (
          <ul className="divide-y">
            {memories.data.map((entry) => (
              <MemoryRow key={entry.mem_no} agentId={id} entry={entry} />
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

function MemoryRow({ agentId, entry }: { agentId: number; entry: MemoryEntry }) {
  const remove = useDeleteMemory(agentId);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  return (
    <li className="flex items-start justify-between gap-4 py-3 first:pt-0 last:pb-0">
      <div className="grid gap-0.5">
        <span className="text-xs text-muted-foreground">#{entry.mem_no}</span>
        <p className="text-sm whitespace-pre-wrap">{entry.content}</p>
      </div>
      <Button
        variant="ghost"
        size="icon-sm"
        aria-label={copy.agents.memories.deleteTitle}
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
            <AlertDialogTitle>{copy.agents.memories.deleteTitle}</AlertDialogTitle>
            <AlertDialogDescription>
              {copy.agents.memories.deleteDescription(entry.mem_no)}
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
                  await remove.mutateAsync(entry.mem_no);
                  setConfirmOpen(false);
                } catch (err) {
                  setError(describeMemoryError(err));
                }
              }}
            >
              {remove.isPending && <LoaderCircle className="animate-spin" />}
              {copy.agents.memories.deleteConfirm}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </li>
  );
}
