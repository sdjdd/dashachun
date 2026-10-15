import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { Check, LoaderCircle } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
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

  return (
    <div className="grid gap-6">
      <AgentEditCard agent={agent.data} />
      <DangerZoneCard agent={agent.data} />
    </div>
  );
}

function AgentEditCard({ agent }: { agent: Agent }) {
  const update = useUpdateAgent();
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
        <CardTitle className="text-base">{copy.agents.editTitle}</CardTitle>
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
        <CardFooter className="justify-between">
          {update.isSuccess && !update.isPending ? (
            <span className="inline-flex items-center gap-1 text-sm text-muted-foreground">
              <Check className="size-4" />
              {copy.agents.saved}
            </span>
          ) : (
            <span />
          )}
          <Button type="submit" disabled={form.formState.isSubmitting}>
            {form.formState.isSubmitting && <LoaderCircle className="animate-spin" />}
            {copy.agents.editSubmit}
          </Button>
        </CardFooter>
      </form>
    </Card>
  );
}

function DangerZoneCard({ agent }: { agent: Agent }) {
  const navigate = useNavigate();
  const remove = useDeleteAgent();
  const [confirmOpen, setConfirmOpen] = useState(false);

  return (
    <Card className="border-destructive/30">
      <CardHeader>
        <CardTitle className="text-base text-destructive">{copy.agents.dangerTitle}</CardTitle>
      </CardHeader>
      <CardContent>
        <Button variant="destructive" onClick={() => setConfirmOpen(true)}>
          {copy.agents.deleteAction}
        </Button>
      </CardContent>
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
