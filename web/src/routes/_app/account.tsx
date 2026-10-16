import { createFileRoute } from "@tanstack/react-router";
import { CheckCircle2, Eye, EyeOff, LoaderCircle, Lock, LogOut, Monitor } from "lucide-react";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { ErrorState, FormError, LoadingState } from "@/components/states";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupInput,
} from "@/components/ui/input-group";
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
  const [show, setShow] = useState(false);
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
            <InputGroup>
              <InputGroupAddon>
                <Lock />
              </InputGroupAddon>
              <InputGroupInput
                id="current_password"
                type={show ? "text" : "password"}
                autoComplete="current-password"
                {...form.register("current_password")}
                aria-invalid={!!errors.current_password}
              />
              <InputGroupAddon align="inline-end">
                <InputGroupButton
                  variant="ghost"
                  size="icon-sm"
                  aria-label={show ? copy.auth.fields.hidePassword : copy.auth.fields.showPassword}
                  aria-pressed={show}
                  onClick={() => setShow((v) => !v)}
                >
                  {show ? <EyeOff /> : <Eye />}
                </InputGroupButton>
              </InputGroupAddon>
            </InputGroup>
            {errors.current_password && (
              <p className="text-sm text-destructive">{errors.current_password.message}</p>
            )}
          </div>
          <div className="grid gap-2">
            <Label htmlFor="new_password">{copy.account.fields.newPassword}</Label>
            <InputGroup>
              <InputGroupAddon>
                <Lock />
              </InputGroupAddon>
              <InputGroupInput
                id="new_password"
                type={show ? "text" : "password"}
                autoComplete="new-password"
                {...form.register("new_password")}
                aria-invalid={!!errors.new_password}
              />
              <InputGroupAddon align="inline-end">
                <InputGroupButton
                  variant="ghost"
                  size="icon-sm"
                  aria-label={show ? copy.auth.fields.hidePassword : copy.auth.fields.showPassword}
                  aria-pressed={show}
                  onClick={() => setShow((v) => !v)}
                >
                  {show ? <EyeOff /> : <Eye />}
                </InputGroupButton>
              </InputGroupAddon>
            </InputGroup>
            {errors.new_password && (
              <p className="text-sm text-destructive">{errors.new_password.message}</p>
            )}
          </div>
          <div className="grid gap-2">
            <Label htmlFor="confirm_password">{copy.account.fields.confirmPassword}</Label>
            <InputGroup>
              <InputGroupAddon>
                <Lock />
              </InputGroupAddon>
              <InputGroupInput
                id="confirm_password"
                type={show ? "text" : "password"}
                autoComplete="new-password"
                {...form.register("confirm_password")}
                aria-invalid={!!errors.confirm_password}
              />
              <InputGroupAddon align="inline-end">
                <InputGroupButton
                  variant="ghost"
                  size="icon-sm"
                  aria-label={show ? copy.auth.fields.hidePassword : copy.auth.fields.showPassword}
                  aria-pressed={show}
                  onClick={() => setShow((v) => !v)}
                >
                  {show ? <EyeOff /> : <Eye />}
                </InputGroupButton>
              </InputGroupAddon>
            </InputGroup>
            {errors.confirm_password && (
              <p className="text-sm text-destructive">{errors.confirm_password.message}</p>
            )}
          </div>
          {errors.root?.message && <FormError message={errors.root.message} />}
          {saved && (
            <p className="flex items-center gap-1.5 text-sm text-muted-foreground">
              <CheckCircle2 className="size-4 text-primary" />
              {copy.account.password.saved}
            </p>
          )}
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
      <Card>
        <CardHeader>
          <CardTitle className="text-base">{copy.account.sessions.title}</CardTitle>
        </CardHeader>
        <CardContent>
          <LoadingState className="py-8" />
        </CardContent>
      </Card>
    );
  }
  if (sessions.isError) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">{copy.account.sessions.title}</CardTitle>
        </CardHeader>
        <CardContent>
          <ErrorState
            message={describeSessionError(sessions.error)}
            onRetry={() => sessions.refetch()}
            className="py-8"
          />
        </CardContent>
      </Card>
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
    <li className="flex flex-wrap items-center justify-between gap-2 py-3 first:pt-0 last:pb-0">
      <div className="flex min-w-0 items-center gap-3">
        <div className="grid size-9 shrink-0 place-items-center rounded-full bg-muted text-muted-foreground">
          <Monitor className="size-4" />
        </div>
        <div className="grid min-w-0 gap-0.5">
          <div className="flex items-center gap-2">
            <span className="truncate text-sm font-medium">{session.ip}</span>
            {session.current && (
              <Badge className="bg-primary/10 text-primary">
                {copy.account.sessions.currentBadge}
              </Badge>
            )}
          </div>
          <p className="truncate text-xs text-muted-foreground" title={session.user_agent}>
            {session.user_agent}
          </p>
          <p className="text-xs text-muted-foreground">
            {copy.account.sessions.lastActive(
              new Date(session.last_active_at).toLocaleString("zh-CN"),
            )}
            {" · "}
            {copy.account.sessions.loggedIn(
              new Date(session.created_at).toLocaleDateString("zh-CN"),
            )}
          </p>
        </div>
      </div>
      {!session.current && (
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label={copy.account.sessions.revoke}
          onClick={() => {
            setError(null);
            setConfirmOpen(true);
          }}
        >
          <LogOut className="size-4" />
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
          {error && <FormError message={error} />}
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
