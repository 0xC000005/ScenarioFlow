import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { Slot } from "radix-ui"

import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "group/badge inline-flex h-5 w-fit shrink-0 items-center justify-center gap-1 overflow-hidden rounded-none border border-transparent px-2 py-0.5 text-xs font-semibold whitespace-nowrap transition-all focus-visible:border-ring  has-data-[icon=inline-end]:pr-1.5 has-data-[icon=inline-start]:pl-1.5 aria-invalid:border-destructive aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 [&>svg]:pointer-events-none [&>svg]:size-3!",
  {
    variants: {
      variant: {
        default: "bg-vg-red text-vg-white [a]:hover:bg-vg-red-deep",
        secondary:
          "border-l-4 border-l-vg-red bg-vg-black text-vg-white [a]:hover:bg-vg-panel",
        destructive:
          "border-vg-red text-vg-red [a]:hover:bg-vg-red [a]:hover:text-vg-white",
        outline:
          "border-vg-black text-vg-black [a]:hover:bg-vg-black [a]:hover:text-vg-white",
        ghost:
          "hover:bg-vg-paper hover:text-vg-black",
        link: "text-vg-black underline decoration-vg-red decoration-2 underline-offset-4",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

function Badge({
  className,
  variant = "default",
  asChild = false,
  ...props
}: React.ComponentProps<"span"> &
  VariantProps<typeof badgeVariants> & { asChild?: boolean }) {
  const Comp = asChild ? Slot.Root : "span"

  return (
    <Comp
      data-slot="badge"
      data-variant={variant}
      className={cn(badgeVariants({ variant }), className)}
      {...props}
    />
  )
}

// eslint-disable-next-line react-refresh/only-export-components -- shadcn style helper is public API
export { Badge, badgeVariants }
