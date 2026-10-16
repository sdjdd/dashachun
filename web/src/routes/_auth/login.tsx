import { Link, createFileRoute, useNavigate } from "@tanstack/react-router";
import { Eye, EyeOff, LoaderCircle, Lock, UserRound } from "lucide-react";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { FormError } from "@/components/states";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupInput,
} from "@/components/ui/input-group";
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
  const [showPassword, setShowPassword] = useState(false);
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
    <Card className="w-full">
      <CardHeader className="items-center text-center">
        <CardTitle className="text-lg">{copy.auth.login.title}</CardTitle>
        <CardDescription>{copy.auth.login.subtitle}</CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={form.handleSubmit(onSubmit)} className="grid gap-4" noValidate>
          <div className="grid gap-2">
            <Label htmlFor="username">{copy.auth.fields.username}</Label>
            <InputGroup>
              <InputGroupAddon>
                <UserRound />
              </InputGroupAddon>
              <InputGroupInput
                id="username"
                {...form.register("username")}
                autoComplete="username"
                autoFocus
                aria-invalid={!!errors.username}
              />
            </InputGroup>
            {errors.username && (
              <p className="text-sm text-destructive">{errors.username.message}</p>
            )}
          </div>
          <div className="grid gap-2">
            <Label htmlFor="password">{copy.auth.fields.password}</Label>
            <InputGroup>
              <InputGroupAddon>
                <Lock />
              </InputGroupAddon>
              <InputGroupInput
                id="password"
                type={showPassword ? "text" : "password"}
                {...form.register("password")}
                autoComplete="current-password"
                aria-invalid={!!errors.password}
              />
              <InputGroupAddon align="inline-end">
                <InputGroupButton
                  variant="ghost"
                  size="icon-xs"
                  aria-label={
                    showPassword ? copy.auth.fields.hidePassword : copy.auth.fields.showPassword
                  }
                  aria-pressed={showPassword}
                  onClick={() => setShowPassword((v) => !v)}
                >
                  {showPassword ? <EyeOff /> : <Eye />}
                </InputGroupButton>
              </InputGroupAddon>
            </InputGroup>
            {errors.password && (
              <p className="text-sm text-destructive">{errors.password.message}</p>
            )}
          </div>
          {errors.root?.message && <FormError message={errors.root.message} />}
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
