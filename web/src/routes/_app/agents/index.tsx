import { Link, createFileRoute, useNavigate } from "@tanstack/react-router";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { LoaderCircle, Plus } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { agentSchema, describeAgentError, useAgents, useCreateAgent } from "@/lib/agents";
import type { Agent, AgentInput } from "@/lib/agents";
import { copy } from "@/lib/copy";

export const Route = createFileRoute("/_app/agents/")({
  component: AgentsPage,
});

function AgentsPage() {
  const agents = useAgents();
  const [createOpen, setCreateOpen] = useState(false);

  return (
    <div className="grid gap-6">
      <div className="flex items-start justify-between gap-4">
        <div className="grid gap-1">
          <h1 className="text-xl font-semibold">{copy.agents.title}</h1>
          <p className="text-sm text-muted-foreground">{copy.agents.description}</p>
        </div>
        <Button onClick={() => setCreateOpen(true)}>
          <Plus />
          {copy.agents.create}
        </Button>
      </div>

      {agents.isPending ? (
        <div className="grid place-items-center py-24 text-muted-foreground">
          <LoaderCircle className="animate-spin" />
        </div>
      ) : agents.isError ? (
        <div className="grid place-items-center gap-3 py-24">
          <p className="text-sm text-destructive">{describeAgentError(agents.error)}</p>
          <Button variant="outline" size="sm" onClick={() => agents.refetch()}>
            {copy.agents.retry}
          </Button>
        </div>
      ) : agents.data.length === 0 ? (
        <div className="grid place-items-center gap-1 py-24 text-center">
          <p className="text-sm text-muted-foreground">{copy.agents.empty}</p>
          <p className="text-xs text-muted-foreground">{copy.agents.emptyHint}</p>
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2">
          {agents.data.map((agent) => (
            <AgentCard key={agent.id} agent={agent} />
          ))}
        </div>
      )}

      <CreateAgentDialog open={createOpen} onOpenChange={setCreateOpen} />
    </div>
  );
}

function AgentCard({ agent }: { agent: Agent }) {
  const date = new Date(agent.created_at).toLocaleDateString("zh-CN");

  return (
    <Link to="/agents/$agentId" params={{ agentId: String(agent.id) }} className="group">
      <Card className="h-full transition-colors group-hover:border-ring">
        <CardHeader>
          <CardTitle className="text-base">{agent.name}</CardTitle>
          <CardDescription>
            {copy.agents.createdAt} {date}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <p className="line-clamp-3 text-sm text-muted-foreground">
            {agent.persona_prompt || copy.agents.fields.personaPlaceholder}
          </p>
        </CardContent>
      </Card>
    </Link>
  );
}

function CreateAgentDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const navigate = useNavigate();
  const create = useCreateAgent();
  const form = useForm<AgentInput>({
    resolver: zodResolver(agentSchema),
    defaultValues: { name: "", persona_prompt: "" },
  });
  const errors = form.formState.errors;

  async function onSubmit(values: AgentInput) {
    form.clearErrors("root");
    try {
      const agent = await create.mutateAsync(values);
      onOpenChange(false);
      navigate({ to: "/agents/$agentId", params: { agentId: String(agent.id) } });
    } catch (err) {
      form.setError("root", { message: describeAgentError(err) });
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{copy.agents.createTitle}</DialogTitle>
        </DialogHeader>
        <form onSubmit={form.handleSubmit(onSubmit)} className="grid gap-4" noValidate>
          <div className="grid gap-2">
            <Label htmlFor="agent-name">{copy.agents.fields.name}</Label>
            <Input
              id="agent-name"
              placeholder={copy.agents.fields.namePlaceholder}
              {...form.register("name")}
              autoFocus
              aria-invalid={!!errors.name}
            />
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
              rows={4}
              placeholder={copy.agents.fields.personaPlaceholder}
              {...form.register("persona_prompt")}
              aria-invalid={!!errors.persona_prompt}
            />
            {errors.persona_prompt && (
              <p className="text-sm text-destructive">{errors.persona_prompt.message}</p>
            )}
          </div>
          {errors.root && <p className="text-sm text-destructive">{errors.root.message}</p>}
          <DialogFooter>
            <DialogClose render={<Button type="button" variant="outline" />}>
              {copy.agents.cancel}
            </DialogClose>
            <Button type="submit" disabled={form.formState.isSubmitting}>
              {form.formState.isSubmitting && <LoaderCircle className="animate-spin" />}
              {copy.agents.createSubmit}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
