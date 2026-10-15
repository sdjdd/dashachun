import { createFileRoute, redirect, useNavigate } from "@tanstack/react-router";
import { LoaderCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { sessionQuery, useLogout, useSession } from "@/lib/auth";
import { copy } from "@/lib/copy";

export const Route = createFileRoute("/")({
  beforeLoad: async ({ context }) => {
    const session = await context.queryClient.query(sessionQuery);
    if (!session) throw redirect({ to: "/login" });
  },
  component: HomePage,
});

function HomePage() {
  const navigate = useNavigate();
  const { data: session } = useSession();
  const logout = useLogout();

  return (
    <div className="grid min-h-svh place-items-center px-4">
      <Card className="w-full max-w-sm">
        <CardHeader>
          <CardTitle className="text-xl">
            {copy.auth.home.greeting}
            {session?.username}
          </CardTitle>
          <CardDescription>
            {session?.role === "admin" ? copy.auth.home.roles.admin : copy.auth.home.roles.user}
            {" · "}
            {copy.auth.home.placeholder}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Button
            variant="outline"
            className="w-full"
            disabled={logout.isPending}
            onClick={() =>
              logout.mutate(undefined, { onSuccess: () => navigate({ to: "/login" }) })
            }
          >
            {logout.isPending && <LoaderCircle className="animate-spin" />}
            {copy.auth.home.logout}
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
