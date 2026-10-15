import { Link, Outlet, createFileRoute } from "@tanstack/react-router";
import { ArrowLeft, LoaderCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { describeAgentError, isAgentNotFound, useAgent } from "@/lib/agents";
import { copy } from "@/lib/copy";

export const Route = createFileRoute("/_app/agents/$agentId")({
  component: AgentDetailLayout,
});

function AgentDetailLayout() {
  const { agentId } = Route.useParams();
  const id = Number(agentId);
  const agent = useAgent(id);

  if (!Number.isSafeInteger(id) || id <= 0 || isAgentNotFound(agent.error)) {
    return <NotFoundView />;
  }
  if (agent.isPending) {
    return (
      <div className="grid place-items-center py-24 text-muted-foreground">
        <LoaderCircle className="animate-spin" />
      </div>
    );
  }
  if (agent.isError) {
    return (
      <div className="grid place-items-center gap-3 py-24">
        <p className="text-sm text-destructive">{describeAgentError(agent.error)}</p>
        <Button variant="outline" size="sm" onClick={() => agent.refetch()}>
          {copy.agents.retry}
        </Button>
      </div>
    );
  }

  const tabLink =
    "border-b-2 border-transparent pb-2 text-sm text-muted-foreground transition-colors hover:text-foreground data-[status=active]:border-foreground data-[status=active]:text-foreground";

  return (
    <div className="grid gap-6">
      <div className="grid gap-1">
        <Link
          to="/agents"
          className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="size-4" />
          {copy.agents.back}
        </Link>
        <h1 className="text-xl font-semibold">{agent.data.name}</h1>
        <p className="text-sm text-muted-foreground">
          {copy.agents.createdAt} {new Date(agent.data.created_at).toLocaleDateString("zh-CN")}
        </p>
      </div>

      <nav className="-mb-px flex gap-6 border-b">
        <Link
          to="/agents/$agentId"
          params={{ agentId }}
          activeOptions={{ exact: true }}
          className={tabLink}
        >
          {copy.agents.tabs.basic}
        </Link>
        <Link to="/agents/$agentId/devices" params={{ agentId }} className={tabLink}>
          {copy.agents.tabs.devices}
        </Link>
        <Link to="/agents/$agentId/memories" params={{ agentId }} className={tabLink}>
          {copy.agents.tabs.memories}
        </Link>
      </nav>

      <Outlet />
    </div>
  );
}

function NotFoundView() {
  return (
    <div className="grid place-items-center gap-3 py-24">
      <p className="text-sm text-muted-foreground">{copy.agents.notFound}</p>
      <Button variant="outline" size="sm" render={<Link to="/agents" />}>
        {copy.agents.back}
      </Button>
    </div>
  );
}
