import { createFileRoute, redirect } from "@tanstack/react-router";
import {
  Brain,
  CheckCircle2,
  Eye,
  EyeOff,
  LoaderCircle,
  Lock,
  Mic,
  Pencil,
  Plug,
  Trash2,
  Volume2,
} from "lucide-react";
import type { ComponentProps } from "react";
import { useEffect, useState } from "react";
import { Controller, useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { EmptyState, ErrorState, FormError, LoadingState } from "@/components/states";
import { Badge } from "@/components/ui/badge";
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
import { Button } from "@/components/ui/button";
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupInput,
} from "@/components/ui/input-group";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { sessionQuery } from "@/lib/auth";
import { copy } from "@/lib/copy";
import {
  audioCaptureSchema,
  describeProviderError,
  describeSettingsError,
  otaSchema,
  providerConfigPayload,
  providerFields,
  providerFormValues,
  providerKinds,
  useActivateProvider,
  useAudioCapture,
  useCreateProvider,
  useDeleteProvider,
  useOtaSettings,
  useProviders,
  useSaveAudioCapture,
  useSaveOtaSettings,
  useSaveVoiceprintSettings,
  useUpdateProvider,
  useVoiceprintSettings,
  voiceprintSchema,
} from "@/lib/settings";
import type {
  AudioCaptureFormValues,
  OtaFormValues,
  ProviderFormValues,
  ProviderRow,
  ProviderType,
  VoiceprintFormValues,
} from "@/lib/settings";

export const Route = createFileRoute("/_app/settings")({
  beforeLoad: async ({ context }) => {
    const session = await context.queryClient.ensureQueryData(sessionQuery);
    if (!session || session.role !== "admin") throw redirect({ to: "/" });
  },
  component: SettingsPage,
});

function SettingsPage() {
  return (
    <div className="grid gap-6">
      <div className="grid gap-1">
        <h1 className="text-xl font-semibold">{copy.settings.title}</h1>
        <p className="text-sm text-muted-foreground">{copy.settings.description}</p>
      </div>
      <ProvidersSection />
      <OtaCard />
      <CaptureCard />
      <VoiceprintCard />
    </div>
  );
}

function ProvidersSection() {
  const providers = useProviders();

  if (providers.isPending) {
    return <LoadingState />;
  }
  if (providers.isError) {
    return (
      <ErrorState
        message={describeSettingsError(providers.error)}
        onRetry={() => providers.refetch()}
      />
    );
  }

  const types: ProviderType[] = ["asr", "llm", "tts"];
  return (
    <div className="grid gap-6">
      {types.map((type) => (
        <ProviderTypeCard
          key={type}
          type={type}
          rows={providers.data.filter((row) => row.type === type)}
        />
      ))}
    </div>
  );
}

const typeIcons = {
  asr: Mic,
  llm: Brain,
  tts: Volume2,
} as Record<ProviderType, typeof Mic>;

function ProviderTypeCard({ type, rows }: { type: ProviderType; rows: ProviderRow[] }) {
  const [actionError, setActionError] = useState<string | null>(null);
  const Icon = typeIcons[type];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Icon className="size-4 text-primary" />
          {copy.settings.providerTypes[type]}
        </CardTitle>
        <CardDescription>{copy.settings.typeDescriptions[type]}</CardDescription>
        <CardAction>
          <ProviderDialog type={type} />
        </CardAction>
      </CardHeader>
      <CardContent>
        {actionError && <FormError className="mb-3" message={actionError} />}
        {rows.length === 0 ? (
          <EmptyState
            icon={Plug}
            text={copy.settings.providers.empty}
            hint={copy.settings.providers.emptyHint}
            className="py-8"
          />
        ) : (
          <ul className="divide-y">
            {rows.map((row) => (
              <ProviderRowItem key={row.id} type={type} row={row} onActionError={setActionError} />
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

function ProviderRowItem({
  type,
  row,
  onActionError,
}: {
  type: ProviderType;
  row: ProviderRow;
  onActionError: (message: string | null) => void;
}) {
  const activate = useActivateProvider();
  const remove = useDeleteProvider();
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const activating = activate.isPending && activate.variables === row.id;

  return (
    <li className="flex flex-wrap items-center justify-between gap-2 py-3 first:pt-0 last:pb-0">
      <div className="grid min-w-0 gap-0.5">
        <div className="flex items-center gap-2">
          <span className="truncate text-sm font-medium">{row.name}</span>
          {row.selected && (
            <Badge className="bg-primary/10 text-primary">{copy.settings.providers.selected}</Badge>
          )}
        </div>
        <p className="text-xs text-muted-foreground">
          {copy.settings.kindLabels[row.kind] ?? row.kind} · {copy.settings.providers.updatedAt}{" "}
          {new Date(row.updated_at).toLocaleDateString("zh-CN")}
        </p>
      </div>
      <div className="flex items-center gap-1">
        {!row.selected && (
          <Button
            variant="outline"
            size="sm"
            disabled={activating}
            onClick={() => {
              onActionError(null);
              activate.mutate(row.id, {
                onError: (err) => onActionError(describeProviderError(err)),
              });
            }}
          >
            {activating && <LoaderCircle className="animate-spin" />}
            {copy.settings.providers.activate}
          </Button>
        )}
        <ProviderDialog type={type} row={row} />
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label={copy.settings.providers.delete}
          disabled={row.selected}
          onClick={() => {
            setError(null);
            setConfirmOpen(true);
          }}
        >
          <Trash2 className="size-4" />
        </Button>
      </div>

      <AlertDialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{copy.settings.providers.deleteTitle}</AlertDialogTitle>
            <AlertDialogDescription>
              {copy.settings.providers.deleteDescription(row.name)}
            </AlertDialogDescription>
          </AlertDialogHeader>
          {error && <FormError message={error} />}
          <AlertDialogFooter>
            <AlertDialogCancel>{copy.agents.cancel}</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              disabled={remove.isPending}
              onClick={async () => {
                setError(null);
                try {
                  await remove.mutateAsync(row.id);
                  setConfirmOpen(false);
                } catch (err) {
                  setError(describeProviderError(err));
                }
              }}
            >
              {remove.isPending && <LoaderCircle className="animate-spin" />}
              {copy.settings.providers.deleteConfirm}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </li>
  );
}

function ProviderDialog({ type, row }: { type: ProviderType; row?: ProviderRow }) {
  const create = useCreateProvider();
  const update = useUpdateProvider();
  const [open, setOpen] = useState(false);
  const kinds = providerKinds(type);
  const form = useForm<ProviderFormValues>({
    defaultValues: providerFormValues(type, row?.kind ?? kinds[0].kind, row),
  });
  const errors = form.formState.errors;
  const kind = form.watch("kind");
  const fields = providerFields(type, kind);

  function switchKind(next: string) {
    const name = form.getValues("name");
    form.reset({ ...providerFormValues(type, next, row), name });
  }

  async function onSubmit(values: ProviderFormValues) {
    form.clearErrors("root");
    try {
      const payload = {
        name: values.name,
        kind: values.kind,
        config: providerConfigPayload(type, values.kind, values),
      };
      if (row) {
        await update.mutateAsync({ id: row.id, ...payload });
      } else {
        await create.mutateAsync({ type, ...payload });
      }
      setOpen(false);
    } catch (err) {
      form.setError("root", {
        message: describeProviderError(err, {
          conflict: copy.settings.providers.errors.nameTaken,
        }),
      });
    }
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        setOpen(next);
        if (next) form.reset(providerFormValues(type, row?.kind ?? kinds[0].kind, row));
      }}
    >
      {row ? (
        <DialogTrigger
          render={
            <Button variant="ghost" size="icon-sm" aria-label={copy.settings.providers.edit} />
          }
        >
          <Pencil className="size-4" />
        </DialogTrigger>
      ) : (
        <DialogTrigger render={<Button size="sm" />}>
          {copy.settings.providers.create}
        </DialogTrigger>
      )}
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>
            {row
              ? copy.settings.providers.editTitle
              : copy.settings.providers.createTitle(copy.settings.providerTypes[type])}
          </DialogTitle>
          <DialogDescription>
            {row ? copy.settings.providerTypes[type] : copy.settings.typeDescriptions[type]}
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={form.handleSubmit(onSubmit)} className="grid gap-4" noValidate>
          <div className="grid gap-2">
            <Label htmlFor="provider-name">{copy.settings.providers.fields.name}</Label>
            <Input
              id="provider-name"
              {...form.register("name", {
                required: copy.settings.providers.validation.nameRequired,
              })}
              aria-invalid={!!errors.name}
            />
            {errors.name && <p className="text-sm text-destructive">{errors.name.message}</p>}
          </div>
          <div className="grid gap-2">
            <Label htmlFor="provider-kind">{copy.settings.providers.fields.kind}</Label>
            <Controller
              control={form.control}
              name="kind"
              render={({ field }) => (
                <Select
                  items={kinds.map((entry) => ({
                    value: entry.kind,
                    label: copy.settings.kindLabels[entry.kind] ?? entry.kind,
                  }))}
                  value={field.value}
                  onValueChange={(value) => {
                    field.onChange(value);
                    if (typeof value === "string" && value !== kind) switchKind(value);
                  }}
                >
                  <SelectTrigger id="provider-kind" className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {kinds.map((entry) => (
                      <SelectItem key={entry.kind} value={entry.kind}>
                        {copy.settings.kindLabels[entry.kind] ?? entry.kind}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            />
          </div>
          {fields.map((spec) => {
            const label = copy.settings.providers.fieldLabels[spec.key] ?? spec.key;
            const inputProps = {
              id: `provider-${spec.key}`,
              placeholder: spec.placeholderKey
                ? copy.settings.providers.placeholders[spec.placeholderKey]
                : undefined,
              ...form.register(spec.key, {
                validate: (value: string) => {
                  const trimmed = (value ?? "").trim();
                  if (spec.required && trimmed === "")
                    return copy.settings.providers.validation.required(label);
                  if (spec.number && trimmed !== "") {
                    if (!/^\d+$/.test(trimmed))
                      return copy.settings.providers.validation.notNumber(label);
                    if (spec.min !== undefined && Number(trimmed) < spec.min)
                      return copy.settings.providers.validation.minNumber(label, spec.min);
                  }
                  return true;
                },
              }),
              "aria-invalid": !!errors[spec.key],
            } satisfies ComponentProps<typeof InputGroupInput>;
            return (
              <div key={spec.key} className="grid gap-2">
                <div className="flex items-baseline justify-between">
                  <Label htmlFor={`provider-${spec.key}`}>{label}</Label>
                  {!spec.required && (
                    <span className="text-xs text-muted-foreground">
                      {copy.agents.fields.personaOptional}
                    </span>
                  )}
                </div>
                {spec.secret ? <SecretField {...inputProps} /> : <Input {...inputProps} />}
                {errors[spec.key] && (
                  <p className="text-sm text-destructive">{errors[spec.key]?.message}</p>
                )}
                {spec.hintKey && (
                  <p className="text-xs text-muted-foreground">
                    {copy.settings.providers.hints[spec.hintKey]}
                  </p>
                )}
              </div>
            );
          })}
          {errors.root?.message && <FormError message={errors.root.message} />}
          <DialogFooter>
            <DialogClose render={<Button variant="outline" type="button" />}>
              {copy.agents.cancel}
            </DialogClose>
            <Button type="submit" disabled={form.formState.isSubmitting}>
              {form.formState.isSubmitting && <LoaderCircle className="animate-spin" />}
              {row ? copy.settings.providers.editSubmit : copy.settings.providers.createSubmit}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

function OtaCard() {
  const ota = useOtaSettings();
  const save = useSaveOtaSettings();
  const [saved, setSaved] = useState(false);
  const form = useForm<OtaFormValues>({
    resolver: zodResolver(otaSchema),
    defaultValues: { websocket_url: "", timezone_offset: "480" },
  });
  const errors = form.formState.errors;

  useEffect(() => {
    if (ota.data) {
      form.reset({
        websocket_url: ota.data.websocket_url ?? "",
        timezone_offset: String(ota.data.timezone_offset),
      });
    }
  }, [ota.data, form]);

  if (ota.isPending) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">{copy.settings.ota.title}</CardTitle>
        </CardHeader>
        <CardContent>
          <LoadingState className="py-8" />
        </CardContent>
      </Card>
    );
  }
  if (ota.isError) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">{copy.settings.ota.title}</CardTitle>
        </CardHeader>
        <CardContent>
          <ErrorState
            message={describeSettingsError(ota.error)}
            onRetry={() => ota.refetch()}
            className="py-8"
          />
        </CardContent>
      </Card>
    );
  }

  async function onSubmit(values: OtaFormValues) {
    form.clearErrors("root");
    setSaved(false);
    try {
      await save.mutateAsync({
        websocket_url: values.websocket_url === "" ? null : values.websocket_url,
        timezone_offset: Number(values.timezone_offset),
      });
      setSaved(true);
    } catch (err) {
      form.setError("root", { message: describeSettingsError(err) });
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{copy.settings.ota.title}</CardTitle>
        <CardDescription>{copy.settings.ota.description}</CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={form.handleSubmit(onSubmit)} className="grid max-w-sm gap-4" noValidate>
          <div className="grid gap-2">
            <Label htmlFor="ota-websocket-url">{copy.settings.ota.wsUrlLabel}</Label>
            <Input
              id="ota-websocket-url"
              placeholder="wss://example.com/ws"
              {...form.register("websocket_url")}
              aria-invalid={!!errors.websocket_url}
            />
            <p className="text-xs text-muted-foreground">{copy.settings.ota.wsUrlHint}</p>
            {errors.websocket_url && (
              <p className="text-sm text-destructive">{errors.websocket_url.message}</p>
            )}
          </div>
          <div className="grid gap-2">
            <Label htmlFor="ota-timezone-offset">{copy.settings.ota.offsetLabel}</Label>
            <Input
              id="ota-timezone-offset"
              inputMode="numeric"
              {...form.register("timezone_offset")}
              aria-invalid={!!errors.timezone_offset}
            />
            <p className="text-xs text-muted-foreground">{copy.settings.ota.offsetHint}</p>
            {errors.timezone_offset && (
              <p className="text-sm text-destructive">{errors.timezone_offset.message}</p>
            )}
          </div>
          {errors.root?.message && <FormError message={errors.root.message} />}
          {saved && (
            <p className="flex items-center gap-1.5 text-sm text-muted-foreground">
              <CheckCircle2 className="size-4 text-primary" />
              {copy.settings.ota.saved}
            </p>
          )}
          <Button
            type="submit"
            disabled={form.formState.isSubmitting}
            className="justify-self-start"
          >
            {form.formState.isSubmitting && <LoaderCircle className="animate-spin" />}
            {copy.settings.ota.submit}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}

function CaptureCard() {
  const capture = useAudioCapture();
  const save = useSaveAudioCapture();
  const [saved, setSaved] = useState(false);
  const form = useForm<AudioCaptureFormValues>({
    resolver: zodResolver(audioCaptureSchema),
    defaultValues: { enabled: false, bucket: "" },
  });
  const errors = form.formState.errors;

  useEffect(() => {
    if (capture.data) {
      form.reset({ enabled: capture.data.enabled, bucket: capture.data.bucket });
    }
  }, [capture.data, form]);

  if (capture.isPending) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">{copy.settings.capture.title}</CardTitle>
        </CardHeader>
        <CardContent>
          <LoadingState className="py-8" />
        </CardContent>
      </Card>
    );
  }
  if (capture.isError) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">{copy.settings.capture.title}</CardTitle>
        </CardHeader>
        <CardContent>
          <ErrorState
            message={describeSettingsError(capture.error)}
            onRetry={() => capture.refetch()}
            className="py-8"
          />
        </CardContent>
      </Card>
    );
  }

  async function onSubmit(values: AudioCaptureFormValues) {
    form.clearErrors("root");
    setSaved(false);
    try {
      await save.mutateAsync({ enabled: values.enabled, bucket: values.bucket });
      setSaved(true);
    } catch (err) {
      form.setError("root", { message: describeSettingsError(err) });
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{copy.settings.capture.title}</CardTitle>
        <CardDescription>{copy.settings.capture.description}</CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={form.handleSubmit(onSubmit)} className="grid max-w-sm gap-4" noValidate>
          <Controller
            control={form.control}
            name="enabled"
            render={({ field }) => (
              <Label className="flex items-center gap-2">
                <Switch checked={field.value} onCheckedChange={field.onChange} />
                {copy.settings.capture.enabledLabel}
              </Label>
            )}
          />
          <div className="grid gap-2">
            <Label htmlFor="capture-bucket">{copy.settings.capture.bucketLabel}</Label>
            <Input
              id="capture-bucket"
              placeholder={copy.settings.capture.bucketPlaceholder}
              {...form.register("bucket")}
              aria-invalid={!!errors.bucket}
            />
            {errors.bucket && <p className="text-sm text-destructive">{errors.bucket.message}</p>}
          </div>
          {errors.root?.message && <FormError message={errors.root.message} />}
          {saved && (
            <p className="flex items-center gap-1.5 text-sm text-muted-foreground">
              <CheckCircle2 className="size-4 text-primary" />
              {copy.settings.capture.saved}
            </p>
          )}
          <Button
            type="submit"
            disabled={form.formState.isSubmitting}
            className="justify-self-start"
          >
            {form.formState.isSubmitting && <LoaderCircle className="animate-spin" />}
            {copy.settings.capture.submit}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}

function VoiceprintCard() {
  const voiceprint = useVoiceprintSettings();
  const save = useSaveVoiceprintSettings();
  const [saved, setSaved] = useState(false);
  const form = useForm<VoiceprintFormValues>({
    resolver: zodResolver(voiceprintSchema),
    defaultValues: { match_threshold: "0.4" },
  });
  const errors = form.formState.errors;

  useEffect(() => {
    if (voiceprint.data) {
      form.reset({ match_threshold: String(voiceprint.data.match_threshold) });
    }
  }, [voiceprint.data, form]);

  if (voiceprint.isPending) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">{copy.settings.voiceprint.title}</CardTitle>
        </CardHeader>
        <CardContent>
          <LoadingState className="py-8" />
        </CardContent>
      </Card>
    );
  }
  if (voiceprint.isError) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">{copy.settings.voiceprint.title}</CardTitle>
        </CardHeader>
        <CardContent>
          <ErrorState
            message={describeSettingsError(voiceprint.error)}
            onRetry={() => voiceprint.refetch()}
            className="py-8"
          />
        </CardContent>
      </Card>
    );
  }

  async function onSubmit(values: VoiceprintFormValues) {
    form.clearErrors("root");
    setSaved(false);
    try {
      await save.mutateAsync({ match_threshold: Number(values.match_threshold) });
      setSaved(true);
    } catch (err) {
      form.setError("root", { message: describeSettingsError(err) });
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{copy.settings.voiceprint.title}</CardTitle>
        <CardDescription>{copy.settings.voiceprint.description}</CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={form.handleSubmit(onSubmit)} className="grid max-w-sm gap-4" noValidate>
          <div className="grid gap-2">
            <Label htmlFor="voiceprint-threshold">{copy.settings.voiceprint.thresholdLabel}</Label>
            <Input
              id="voiceprint-threshold"
              inputMode="decimal"
              {...form.register("match_threshold")}
              aria-invalid={!!errors.match_threshold}
            />
            <p className="text-xs text-muted-foreground">
              {copy.settings.voiceprint.thresholdHint}
            </p>
            {errors.match_threshold && (
              <p className="text-sm text-destructive">{errors.match_threshold.message}</p>
            )}
          </div>
          {errors.root?.message && <FormError message={errors.root.message} />}
          {saved && (
            <p className="flex items-center gap-1.5 text-sm text-muted-foreground">
              <CheckCircle2 className="size-4 text-primary" />
              {copy.settings.voiceprint.saved}
            </p>
          )}
          <Button
            type="submit"
            disabled={form.formState.isSubmitting}
            className="justify-self-start"
          >
            {form.formState.isSubmitting && <LoaderCircle className="animate-spin" />}
            {copy.settings.voiceprint.submit}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}

function SecretField(props: ComponentProps<typeof InputGroupInput>) {
  const [show, setShow] = useState(false);
  return (
    <InputGroup>
      <InputGroupAddon>
        <Lock />
      </InputGroupAddon>
      <InputGroupInput {...props} type={show ? "text" : "password"} autoComplete="off" />
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
  );
}
