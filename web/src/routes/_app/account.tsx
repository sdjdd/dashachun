import { createFileRoute } from "@tanstack/react-router";
import { LoaderCircle } from "lucide-react";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
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
import {
  changePasswordSchema,
  describePasswordError,
  describeSessionError,
  useChangePassword,
  useRevokeSession,
  useSessions,
} from "@/lib/account";
import type { ChangePasswordValues, DeviceSession } from "@/lib/account";
import { copy } from "@/lib/copy";

export const Route = createFileRoute("/_app/account")({
  component: AccountPage,
});

function AccountPage() {
  return (
    <div className="grid gap-6">
      <div className="grid gap-1">
        <h1 className="text-xl font-semibold">{copy.account.title}</h1>
        <p className="text-sm text-muted-foreground">{copy.account.description}</p>
      </div>
      <PasswordCard />
      <SessionsCard />
    </div>
  );
}

function PasswordCard() {
  const change = useChangePassword();
  const [saved, setSaved] = useState(false);
  const form = useForm<ChangePasswordValues>({
    resolver: zodResolver(changePasswordSchema),
    defaultValues: { current_password: "", new_password: "", confirm_password: "" },
  });
  const errors = form.formState.errors;

  async function onSubmit(values: ChangePasswordValues) {
    form.clearErrors("root");
    setSaved(false);
    try {
      await change.mutateAsync(values);
      form.reset();
      setSaved(true);
    } catch (err) {
      form.setError("root", { message: describePasswordError(err) });
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{copy.account.password.title}</CardTitle>
        <CardDescription>{copy.account.password.description}</CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={form.handleSubmit(onSubmit)} className="grid max-w-sm gap-4" noValidate>
          <div className="grid gap-2">
            <Label htmlFor="current_password">{copy.account.fields.currentPassword}</Label>
            <Input
              id="current_password"
              type="password"
              autoComplete="current-password"
              {...form.register("current_password")}
              aria-invalid={!!errors.current_password}
            />
            {errors.current_password && (
              <p className="text-sm text-destructive">{errors.current_password.message}</p>
            )}
          </div>
          <div className="grid gap-2">
            <Label htmlFor="new_password">{copy.account.fields.newPassword}</Label>
            <Input
              id="new_password"
              type="password"
              autoComplete="new-password"
              {...form.register("new_password")}
              aria-invalid={!!errors.new_password}
            />
            {errors.new_password && (
              <p className="text-sm text-destructive">{errors.new_password.message}</p>
            )}
          </div>
          <div className="grid gap-2">
            <Label htmlFor="confirm_password">{copy.account.fields.confirmPassword}</Label>
            <Input
              id="confirm_password"
              type="password"
              autoComplete="new-password"
              {...form.register("confirm_password")}
              aria-invalid={!!errors.confirm_password}
            />
            {errors.confirm_password && (
              <p className="text-sm text-destructive">{errors.confirm_password.message}</p>
            )}
          </div>
          {errors.root && <p className="text-sm text-destructive">{errors.root.message}</p>}
          {saved && <p className="text-sm text-muted-foreground">{copy.account.password.saved}</p>}
          <Button
            type="submit"
            disabled={form.formState.isSubmitting}
            className="justify-self-start"
          >
            {form.formState.isSubmitting && <LoaderCircle className="animate-spin" />}
            {copy.account.password.submit}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}

function SessionsCard() {
  const sessions = useSessions();

  if (sessions.isPending) {
    return (
      <div className="grid place-items-center py-16 text-muted-foreground">
        <LoaderCircle className="animate-spin" />
      </div>
    );
  }
  if (sessions.isError) {
    return (
      <div className="grid place-items-center gap-3 py-16">
        <p className="text-sm text-destructive">{describeSessionError(sessions.error)}</p>
        <Button variant="outline" size="sm" onClick={() => sessions.refetch()}>
          {copy.agents.retry}
        </Button>
      </div>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{copy.account.sessions.title}</CardTitle>
        <CardDescription>{copy.account.sessions.description}</CardDescription>
      </CardHeader>
      <CardContent>
        <ul className="divide-y">
          {sessions.data.map((session) => (
            <SessionRow key={session.id} session={session} />
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}

function SessionRow({ session }: { session: DeviceSession }) {
  const revoke = useRevokeSession();
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  return (
    <li className="flex items-start justify-between gap-4 py-3 first:pt-0 last:pb-0">
      <div className="grid min-w-0 gap-0.5">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">{session.ip}</span>
          {session.current && (
            <span className="rounded-full bg-primary/10 px-2 py-0.5 text-xs text-primary">
              {copy.account.sessions.currentBadge}
            </span>
          )}
        </div>
        <p className="max-w-xl truncate text-xs text-muted-foreground" title={session.user_agent}>
          {session.user_agent}
        </p>
        <p className="text-xs text-muted-foreground">
          {copy.account.sessions.lastActive(
            new Date(session.last_active_at).toLocaleString("zh-CN"),
          )}
          {" · "}
          {copy.account.sessions.loggedIn(new Date(session.created_at).toLocaleDateString("zh-CN"))}
        </p>
      </div>
      {!session.current && (
        <Button
          variant="ghost"
          size="sm"
          onClick={() => {
            setError(null);
            setConfirmOpen(true);
          }}
        >
          {copy.account.sessions.revoke}
        </Button>
      )}

      <AlertDialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{copy.account.sessions.revokeTitle}</AlertDialogTitle>
            <AlertDialogDescription>
              {copy.account.sessions.revokeDescription}
            </AlertDialogDescription>
          </AlertDialogHeader>
          {error && <p className="text-sm text-destructive">{error}</p>}
          <AlertDialogFooter>
            <AlertDialogCancel>{copy.agents.cancel}</AlertDialogCancel>
            <AlertDialogAction
              disabled={revoke.isPending}
              onClick={async () => {
                setError(null);
                try {
                  await revoke.mutateAsync(session.id);
                  setConfirmOpen(false);
                } catch (err) {
                  setError(describeSessionError(err));
                }
              }}
            >
              {revoke.isPending && <LoaderCircle className="animate-spin" />}
              {copy.account.sessions.revokeConfirm}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </li>
  );
}
