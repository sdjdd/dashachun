import type { LucideIcon } from "lucide-react";
import { LoaderCircle, RotateCw, TriangleAlert } from "lucide-react";
import type { ReactNode } from "react";
import { cn } from "cn";
import { Button } from "@/components/ui/button";
import { copy } from "@/lib/copy";

export function LoadingState({ className }: { className?: string }) {
  return (
    <div className={cn("grid place-items-center py-16 text-muted-foreground", className)}>
      <LoaderCircle className="size-6 animate-spin" />
    </div>
  );
}

export function ErrorState({
  message,
  onRetry,
  className,
}: {
  message: string;
  onRetry: () => void;
  className?: string;
}) {
  return (
    <div className={cn("grid place-items-center gap-3 py-16 text-center", className)}>
      <div className="grid size-12 place-items-center rounded-full bg-destructive/10">
        <TriangleAlert className="size-6 text-destructive" />
      </div>
      <p className="text-sm text-destructive">{message}</p>
      <Button variant="outline" size="sm" onClick={onRetry}>
        <RotateCw className="size-4" />
        {copy.common.retry}
      </Button>
    </div>
  );
}

export function FormError({ message, className }: { message: string; className?: string }) {
  return (
    <div
      role="alert"
      className={cn(
        "flex items-start gap-2 rounded-lg bg-destructive/10 px-3 py-2.5 text-sm text-destructive",
        className,
      )}
    >
      <TriangleAlert className="mt-0.5 size-4 shrink-0" />
      <span>{message}</span>
    </div>
  );
}

export function EmptyState({
  icon: Icon,
  text,
  hint,
  action,
  className,
}: {
  icon: LucideIcon;
  text: string;
  hint?: string;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("grid place-items-center gap-1 py-12 text-center", className)}>
      <div className="mb-2 grid size-12 place-items-center rounded-full bg-muted">
        <Icon className="size-6 text-muted-foreground" />
      </div>
      <p className="text-sm text-muted-foreground">{text}</p>
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
      {action && <div className="mt-3">{action}</div>}
    </div>
  );
}
