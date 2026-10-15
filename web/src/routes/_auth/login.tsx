import { Link, createFileRoute, useNavigate } from "@tanstack/react-router";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { LoaderCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { credentialsSchema, describeAuthError, useLogin } from "@/lib/auth";
import type { Credentials } from "@/lib/auth";
import { copy } from "@/lib/copy";

export const Route = createFileRoute("/_auth/login")({
  component: LoginPage,
});

function LoginPage() {
  const navigate = useNavigate();
  const login = useLogin();
  const form = useForm<Credentials>({
    resolver: zodResolver(credentialsSchema),
    defaultValues: { username: "", password: "" },
  });
  const errors = form.formState.errors;

  async function onSubmit(values: Credentials) {
    form.clearErrors("root");
    try {
      await login.mutateAsync(values);
      navigate({ to: "/" });
    } catch (err) {
      form.setError("root", { message: describeAuthError(err) });
    }
  }

  return (
    <Card className="w-full max-w-sm">
      <CardHeader>
        <CardTitle className="text-xl">{copy.brand.name}</CardTitle>
        <CardDescription>{copy.auth.login.subtitle}</CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={form.handleSubmit(onSubmit)} className="grid gap-4" noValidate>
          <div className="grid gap-2">
            <Label htmlFor="username">{copy.auth.fields.username}</Label>
            <Input
              id="username"
              {...form.register("username")}
              autoComplete="username"
              autoFocus
              aria-invalid={!!errors.username}
            />
            {errors.username && (
              <p className="text-sm text-destructive">{errors.username.message}</p>
            )}
          </div>
          <div className="grid gap-2">
            <Label htmlFor="password">{copy.auth.fields.password}</Label>
            <Input
              id="password"
              type="password"
              {...form.register("password")}
              autoComplete="current-password"
              aria-invalid={!!errors.password}
            />
            {errors.password && (
              <p className="text-sm text-destructive">{errors.password.message}</p>
            )}
          </div>
          {errors.root && <p className="text-sm text-destructive">{errors.root.message}</p>}
          <Button type="submit" disabled={form.formState.isSubmitting}>
            {form.formState.isSubmitting && <LoaderCircle className="animate-spin" />}
            {copy.auth.login.submit}
          </Button>
        </form>
      </CardContent>
      <CardFooter className="justify-center text-sm text-muted-foreground">
        {copy.auth.login.switchPrompt}
        <Link to="/register" className="ms-1 font-medium underline-offset-4 hover:underline">
          {copy.auth.login.switchLink}
        </Link>
      </CardFooter>
    </Card>
  );
}
