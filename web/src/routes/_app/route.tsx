import { Link, Outlet, createFileRoute, redirect, useNavigate } from "@tanstack/react-router";
import { LoaderCircle } from "lucide-react";
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
        <div className="mx-auto flex h-14 max-w-5xl items-center gap-6 px-4">
          <Link to="/" className="text-sm font-semibold">
            {copy.brand.name}
          </Link>
          <nav className="flex items-center gap-4 text-sm">
            <Link
              to="/agents"
              className="text-muted-foreground"
              activeProps={{ className: "font-medium text-foreground" }}
            >
              {copy.agents.nav}
            </Link>
            <Link
              to="/account"
              className="text-muted-foreground"
              activeProps={{ className: "font-medium text-foreground" }}
            >
              {copy.account.nav}
            </Link>
          </nav>
          <div className="ms-auto flex items-center gap-3 text-sm text-muted-foreground">
            <span>{session?.username}</span>
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
