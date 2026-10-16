import { Link, Outlet, createFileRoute } from "@tanstack/react-router";
import {
  ArrowLeft,
  AudioWaveform,
  Bot,
  Brain,
  CircleUserRound,
  MessagesSquare,
  Smartphone,
} from "lucide-react";
import { ErrorState, LoadingState } from "@/components/states";
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
    return <LoadingState />;
  }
  if (agent.isError) {
    return <ErrorState message={describeAgentError(agent.error)} onRetry={() => agent.refetch()} />;
  }

  const tabLink =
    "inline-flex items-center gap-1.5 border-b-2 border-transparent pb-2 text-sm whitespace-nowrap text-muted-foreground transition-colors hover:text-foreground data-[status=active]:border-foreground data-[status=active]:text-foreground";

  const tabs = [
    {
      to: "/agents/$agentId",
      exact: true,
      icon: CircleUserRound,
      label: copy.agents.tabs.basic,
    },
    {
      to: "/agents/$agentId/devices",
      exact: false,
      icon: Smartphone,
      label: copy.agents.tabs.devices,
    },
    {
      to: "/agents/$agentId/memories",
      exact: false,
      icon: Brain,
      label: copy.agents.tabs.memories,
    },
    {
      to: "/agents/$agentId/conversations",
      exact: false,
      icon: MessagesSquare,
      label: copy.agents.tabs.conversations,
    },
    {
      to: "/agents/$agentId/speakers",
      exact: false,
      icon: AudioWaveform,
      label: copy.agents.tabs.speakers,
    },
  ];

  return (
    <div className="grid gap-6">
      <div className="grid gap-1">
        <Link
          to="/agents"
          className="inline-flex items-center gap-1 text-sm text-muted-foreground transition-colors hover:text-foreground"
        >
          <ArrowLeft className="size-4" />
          {copy.agents.back}
        </Link>
        <div className="mt-1 flex items-center gap-3">
          <div className="grid size-10 shrink-0 place-items-center rounded-full bg-primary/10 text-primary">
            <Bot className="size-5" />
          </div>
          <div className="min-w-0">
            <h1 className="truncate text-xl font-semibold">{agent.data.name}</h1>
            <p className="text-sm text-muted-foreground">
              {copy.agents.createdAt} {new Date(agent.data.created_at).toLocaleDateString("zh-CN")}
            </p>
          </div>
        </div>
      </div>

      <div className="relative -mx-4 sm:mx-0">
        <div className="absolute inset-x-0 bottom-0 h-px bg-border" />
        <nav className="relative flex gap-5 overflow-x-auto px-4 sm:gap-6 sm:px-0">
          {tabs.map((tab) => (
            <Link
              key={tab.to}
              to={tab.to}
              params={{ agentId }}
              activeOptions={{ exact: tab.exact }}
              className={tabLink}
            >
              <tab.icon className="size-4" />
              {tab.label}
            </Link>
          ))}
        </nav>
      </div>

      <Outlet />
    </div>
  );
}

function NotFoundView() {
  return (
    <div className="grid place-items-center gap-3 py-24 text-center">
      <div className="grid size-12 place-items-center rounded-full bg-muted">
        <Bot className="size-6 text-muted-foreground" />
      </div>
      <p className="text-sm text-muted-foreground">{copy.agents.notFound}</p>
      <Button variant="outline" size="sm" render={<Link to="/agents" />}>
        {copy.agents.back}
      </Button>
    </div>
  );
}
