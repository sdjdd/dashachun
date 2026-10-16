import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { Check, LoaderCircle, PenLine, Trash2 } from "lucide-react";
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
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  agentSchema,
  describeAgentError,
  useAgent,
  useDeleteAgent,
  useUpdateAgent,
} from "@/lib/agents";
import type { Agent, AgentInput } from "@/lib/agents";
import { copy } from "@/lib/copy";

export const Route = createFileRoute("/_app/agents/$agentId/")({
  component: BasicTab,
});

function BasicTab() {
  const { agentId } = Route.useParams();
  const agent = useAgent(Number(agentId));

  if (!agent.data) {
    return null;
  }

  return <AgentEditCard agent={agent.data} />;
}

function AgentEditCard({ agent }: { agent: Agent }) {
  const navigate = useNavigate();
  const update = useUpdateAgent();
  const remove = useDeleteAgent();
  const [confirmOpen, setConfirmOpen] = useState(false);
  const form = useForm<AgentInput>({
    resolver: zodResolver(agentSchema),
    defaultValues: { name: agent.name, persona_prompt: agent.persona_prompt },
  });
  const errors = form.formState.errors;

  async function onSubmit(values: AgentInput) {
    form.clearErrors("root");
    try {
      await update.mutateAsync({ id: agent.id, input: values });
    } catch (err) {
      form.setError("root", { message: describeAgentError(err) });
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <PenLine className="size-4 text-muted-foreground" />
          {copy.agents.editTitle}
        </CardTitle>
      </CardHeader>
      <form onSubmit={form.handleSubmit(onSubmit)} className="grid gap-4" noValidate>
        <CardContent className="grid gap-4">
          <div className="grid gap-2">
            <Label htmlFor="agent-name">{copy.agents.fields.name}</Label>
            <Input id="agent-name" {...form.register("name")} aria-invalid={!!errors.name} />
            {errors.name && <p className="text-sm text-destructive">{errors.name.message}</p>}
          </div>
          <div className="grid gap-2">
            <div className="flex items-baseline justify-between">
              <Label htmlFor="agent-persona">{copy.agents.fields.persona}</Label>
              <span className="text-xs text-muted-foreground">
                {copy.agents.fields.personaOptional}
              </span>
            </div>
            <Textarea
              id="agent-persona"
              rows={10}
              placeholder={copy.agents.fields.personaPlaceholder}
              {...form.register("persona_prompt")}
              aria-invalid={!!errors.persona_prompt}
            />
            {errors.persona_prompt && (
              <p className="text-sm text-destructive">{errors.persona_prompt.message}</p>
            )}
          </div>
          {errors.root && <p className="text-sm text-destructive">{errors.root.message}</p>}
        </CardContent>
        <CardFooter className="justify-between gap-2">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="-ml-2 text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
            onClick={() => setConfirmOpen(true)}
          >
            <Trash2 className="size-4" />
            {copy.agents.deleteAction}
          </Button>
          <div className="flex items-center gap-3">
            {update.isSuccess && !update.isPending && (
              <span className="inline-flex items-center gap-1 text-sm text-muted-foreground">
                <Check className="size-4" />
                {copy.agents.saved}
              </span>
            )}
            <Button type="submit" disabled={form.formState.isSubmitting}>
              {form.formState.isSubmitting && <LoaderCircle className="animate-spin" />}
              {copy.agents.editSubmit}
            </Button>
          </div>
        </CardFooter>
      </form>
      <AlertDialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{copy.agents.deleteTitle}</AlertDialogTitle>
            <AlertDialogDescription>
              {copy.agents.deleteDescription(agent.name)}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{copy.agents.cancel}</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => {
                setConfirmOpen(false);
                remove.mutate(agent.id, { onSuccess: () => navigate({ to: "/agents" }) });
              }}
            >
              {copy.agents.deleteConfirm}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </Card>
  );
}
