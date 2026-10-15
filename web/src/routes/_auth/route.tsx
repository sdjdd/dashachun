import { Outlet, createFileRoute, redirect } from "@tanstack/react-router";
import { sessionQuery } from "@/lib/auth";

export const Route = createFileRoute("/_auth")({
  beforeLoad: async ({ context }) => {
    const session = await context.queryClient.query(sessionQuery);
    if (session) throw redirect({ to: "/" });
  },
  component: () => (
    <div className="grid min-h-svh place-items-center bg-muted/40 px-4">
      <Outlet />
    </div>
  ),
});
