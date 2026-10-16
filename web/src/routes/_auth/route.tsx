import { Outlet, createFileRoute, redirect } from "@tanstack/react-router";
import { AudioLines } from "lucide-react";
import { sessionQuery } from "@/lib/auth";
import { copy } from "@/lib/copy";

export const Route = createFileRoute("/_auth")({
  beforeLoad: async ({ context }) => {
    const session = await context.queryClient.query(sessionQuery);
    if (session) throw redirect({ to: "/" });
  },
  component: () => (
    <div className="grid min-h-svh place-items-center bg-muted/40 px-4">
      <div className="grid w-full max-w-sm justify-items-center gap-6 animate-in fade-in slide-in-from-bottom-2 duration-500">
        <div className="grid justify-items-center gap-3">
          <div className="grid size-12 place-items-center rounded-xl bg-primary text-primary-foreground shadow-sm">
            <AudioLines className="size-6" />
          </div>
          <p className="font-heading text-lg font-semibold tracking-tight">{copy.brand.name}</p>
        </div>
        <Outlet />
      </div>
    </div>
  ),
});
