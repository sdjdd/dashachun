import { createFileRoute } from "@tanstack/react-router";
import { Controller, useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { LoaderCircle, Smartphone } from "lucide-react";
import { useState } from "react";
import { EmptyState, ErrorState, LoadingState } from "@/components/states";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { InputOTP, InputOTPGroup, InputOTPSlot } from "@/components/ui/input-otp";
import { Label } from "@/components/ui/label";
import {
  describeDeviceBindError,
  describeDeviceError,
  describeUnbindError,
  deviceCodeSchema,
  useBindDevice,
  useDevices,
  useUnbindDevice,
} from "@/lib/devices";
import type { Device, DeviceCodeInput } from "@/lib/devices";
import { copy } from "@/lib/copy";

export const Route = createFileRoute("/_app/agents/$agentId/devices")({
  component: DevicesTab,
});

function DevicesTab() {
  const { agentId } = Route.useParams();
  const id = Number(agentId);
  const devices = useDevices(id);

  if (devices.isPending) {
    return <LoadingState />;
  }
  if (devices.isError) {
    return (
      <ErrorState message={describeDeviceError(devices.error)} onRetry={() => devices.refetch()} />
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Smartphone className="size-4 text-muted-foreground" />
          {copy.agents.devices.listTitle}
        </CardTitle>
        <CardAction>
          <BindDialog agentId={id} />
        </CardAction>
      </CardHeader>
      <CardContent>
        {devices.data.length === 0 ? (
          <EmptyState
            icon={Smartphone}
            text={copy.agents.devices.empty}
            hint={copy.agents.devices.emptyHint}
          />
        ) : (
          <ul className="divide-y">
            {devices.data.map((device) => (
              <DeviceRow key={device.client_id} agentId={id} device={device} />
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

function BindDialog({ agentId }: { agentId: number }) {
  const bind = useBindDevice(agentId);
  const [open, setOpen] = useState(false);
  const form = useForm<DeviceCodeInput>({
    resolver: zodResolver(deviceCodeSchema),
    defaultValues: { code: "" },
  });
  const errors = form.formState.errors;

  async function onSubmit(values: DeviceCodeInput) {
    form.clearErrors("root");
    try {
      await bind.mutateAsync(values);
      setOpen(false);
    } catch (err) {
      form.setError("root", { message: describeDeviceBindError(err) });
    }
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        setOpen(next);
        if (next) {
          form.reset({ code: "" });
        }
      }}
    >
      <DialogTrigger render={<Button size="sm" />}>{copy.agents.devices.bindTitle}</DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{copy.agents.devices.bindTitle}</DialogTitle>
          <DialogDescription>{copy.agents.devices.bindDescription}</DialogDescription>
        </DialogHeader>
        <form onSubmit={form.handleSubmit(onSubmit)} className="grid gap-4" noValidate>
          <div className="grid justify-items-center gap-2">
            <Label htmlFor="device-code">{copy.agents.devices.codeLabel}</Label>
            <Controller
              control={form.control}
              name="code"
              render={({ field }) => (
                <InputOTP
                  maxLength={6}
                  value={field.value}
                  onChange={field.onChange}
                  aria-invalid={!!errors.code}
                >
                  <InputOTPGroup>
                    {[0, 1, 2, 3, 4, 5].map((index) => (
                      <InputOTPSlot key={index} index={index} />
                    ))}
                  </InputOTPGroup>
                </InputOTP>
              )}
            />
            {errors.code && <p className="text-sm text-destructive">{errors.code.message}</p>}
            {errors.root && <p className="text-sm text-destructive">{errors.root.message}</p>}
          </div>
          <DialogFooter>
            <DialogClose render={<Button variant="outline" type="button" />}>
              {copy.agents.cancel}
            </DialogClose>
            <Button type="submit" disabled={form.formState.isSubmitting}>
              {form.formState.isSubmitting && <LoaderCircle className="animate-spin" />}
              {copy.agents.devices.bindSubmit}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

function DeviceRow({ agentId, device }: { agentId: number; device: Device }) {
  const unbind = useUnbindDevice(agentId);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const meta = [device.board_type, device.device_id].filter(Boolean).join(" · ");

  return (
    <li className="flex flex-wrap items-center justify-between gap-2 py-3 first:pt-0 last:pb-0">
      <div className="flex min-w-0 items-center gap-3">
        <div className="grid size-9 shrink-0 place-items-center rounded-full bg-muted text-muted-foreground">
          <Smartphone className="size-4" />
        </div>
        <div className="grid min-w-0 gap-0.5">
          <span className="truncate font-mono text-sm">{device.client_id}</span>
          <span className="truncate text-xs text-muted-foreground">{meta || "—"}</span>
        </div>
      </div>
      <div className="flex items-center gap-3">
        {device.activated_at && (
          <span className="text-xs text-muted-foreground">
            {copy.agents.devices.activatedAt}{" "}
            {new Date(device.activated_at).toLocaleDateString("zh-CN")}
          </span>
        )}
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            setError(null);
            setConfirmOpen(true);
          }}
        >
          {copy.agents.devices.unbind}
        </Button>
      </div>

      <AlertDialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{copy.agents.devices.unbindTitle}</AlertDialogTitle>
            <AlertDialogDescription>
              {copy.agents.devices.unbindDescription(device.client_id)}
            </AlertDialogDescription>
          </AlertDialogHeader>
          {error && <p className="text-sm text-destructive">{error}</p>}
          <AlertDialogFooter>
            <AlertDialogCancel>{copy.agents.cancel}</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              disabled={unbind.isPending}
              onClick={async () => {
                setError(null);
                try {
                  await unbind.mutateAsync(device.client_id);
                  setConfirmOpen(false);
                } catch (err) {
                  setError(describeUnbindError(err));
                }
              }}
            >
              {unbind.isPending && <LoaderCircle className="animate-spin" />}
              {copy.agents.devices.unbindConfirm}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </li>
  );
}
