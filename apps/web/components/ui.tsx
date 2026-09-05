"use client";
import { cn } from "@/lib/utils";

export function Card({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("bg-white rounded-2xl border border-zinc-200 shadow-sm", className)} {...props} />;
}
export function CardHeader({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("px-5 py-3.5 border-b border-zinc-100 flex items-center justify-between", className)} {...props} />;
}
export function CardTitle({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("font-bold text-sm tracking-tight", className)} {...props} />;
}
export function Badge({ className, ...props }: React.HTMLAttributes<HTMLSpanElement>) {
  return <span className={cn("inline-flex items-center px-2.5 py-1 rounded-full text-xs font-bold mono", className)} {...props} />;
}
export function Button({
  variant = "default",
  className,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "default" | "ghost" | "outline" }) {
  const base = "inline-flex items-center justify-center rounded-xl text-sm font-bold transition disabled:opacity-40 disabled:cursor-not-allowed";
  const variants: Record<string, string> = {
    default: "bg-zinc-900 text-white hover:bg-black px-4 py-2.5",
    ghost: "hover:bg-zinc-100 px-3 py-1.5",
    outline: "border border-zinc-300 bg-white hover:bg-zinc-50 px-3 py-1.5",
  };
  return <button className={cn(base, variants[variant], className)} {...props} />;
}
