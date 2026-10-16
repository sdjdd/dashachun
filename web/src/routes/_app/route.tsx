import { Link, Outlet, createFileRoute, redirect, useNavigate } from "@tanstack/react-router";
import { Bot, LoaderCircle, Settings, UserRound } from "lucide-react";
import { Button } from "@/components/ui/button";
import { sessionQuery, useLogout, useSession } from "@/lib/auth";
import { copy } from "@/lib/copy";

export const Route = createFileRoute("/_app")({
  beforeLoad: async ({ context }) => {
    const session = await context.queryClient.query(sessionQuery);
    if (!session) throw redirect({ to: "/login" });
  },
  component: AppLayout,
});

function AppLayout() {
  const navigate = useNavigate();
  const { data: session } = useSession();
  const logout = useLogout();

  return (
    <div className="min-h-svh bg-muted/40">
      <header className="border-b bg-background">
        <div className="mx-auto flex h-14 max-w-5xl items-center gap-3 px-4 sm:gap-6">
          <Link to="/" className="text-sm font-semibold">
            {copy.brand.name}
          </Link>
          <nav className="flex items-center gap-1 text-sm sm:gap-2">
            <Link
              to="/agents"
              aria-label={copy.agents.nav}
              className="inline-flex items-center gap-1.5 rounded-md px-2 py-1.5 text-muted-foreground transition-colors hover:text-foreground"
              activeProps={{ className: "font-medium text-foreground" }}
            >
              <Bot className="size-4" />
              <span className="hidden sm:inline">{copy.agents.nav}</span>
            </Link>
            <Link
              to="/account"
              aria-label={copy.account.nav}
              className="inline-flex items-center gap-1.5 rounded-md px-2 py-1.5 text-muted-foreground transition-colors hover:text-foreground"
              activeProps={{ className: "font-medium text-foreground" }}
            >
              <UserRound className="size-4" />
              <span className="hidden sm:inline">{copy.account.nav}</span>
            </Link>
            {session?.role === "admin" && (
              <Link
                to="/settings"
                aria-label={copy.settings.nav}
                className="inline-flex items-center gap-1.5 rounded-md px-2 py-1.5 text-muted-foreground transition-colors hover:text-foreground"
                activeProps={{ className: "font-medium text-foreground" }}
              >
                <Settings className="size-4" />
                <span className="hidden sm:inline">{copy.settings.nav}</span>
              </Link>
            )}
          </nav>
          <div className="ms-auto flex items-center gap-2 text-sm text-muted-foreground sm:gap-3">
            <span className="hidden sm:block">{session?.username}</span>
            <Button
              variant="ghost"
              size="sm"
              disabled={logout.isPending}
              onClick={() =>
                logout.mutate(undefined, { onSuccess: () => navigate({ to: "/login" }) })
              }
            >
              {logout.isPending && <LoaderCircle className="animate-spin" />}
              {copy.shell.logout}
            </Button>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-5xl px-4 py-6">
        <Outlet />
      </main>
    </div>
  );
}
