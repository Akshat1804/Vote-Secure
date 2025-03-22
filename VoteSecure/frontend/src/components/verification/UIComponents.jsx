import React from "react";
import { AlertCircle, AlertTriangle, CheckCircle2, Info } from 'lucide-react';

// Simple utility function for combining class names without dependencies
export const cn = (...classes) => {
  return classes
    .filter(Boolean)
    .join(" ")
    .replace(/border-border/g, "border");
};

// UI Components
export const Button = React.forwardRef(
  ({ className, variant = "default", size = "default", asChild = false, ...props }, ref) => {
    const Comp = asChild ? "button" : "button";

    let variantClasses = "";
    if (variant === "default") variantClasses = "bg-primary text-primary-foreground hover:bg-primary/90";
    else if (variant === "destructive")
      variantClasses = "bg-destructive text-destructive-foreground hover:bg-destructive/90";
    else if (variant === "outline")
      variantClasses = "border border-input bg-background hover:bg-accent hover:text-accent-foreground";
    else if (variant === "secondary") variantClasses = "bg-secondary text-secondary-foreground hover:bg-secondary/80";
    else if (variant === "ghost") variantClasses = "hover:bg-accent hover:text-accent-foreground";
    else if (variant === "link") variantClasses = "text-primary underline-offset-4 hover:underline";

    let sizeClasses = "";
    if (size === "default") sizeClasses = "h-10 px-4 py-2";
    else if (size === "sm") sizeClasses = "h-9 rounded-md px-3";
    else if (size === "lg") sizeClasses = "h-11 rounded-md px-8";
    else if (size === "icon") sizeClasses = "h-10 w-10";

    if (asChild) {
      return <React.Fragment ref={ref} {...props} />;
    }

    return (
      <Comp
        className={cn(
          "inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
          variantClasses,
          sizeClasses,
          className,
        )}
        ref={ref}
        {...props}
      />
    );
  },
);
Button.displayName = "Button";

export const Card = React.forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("rounded-lg border bg-card text-card-foreground shadow-sm", className)} {...props} />
));
Card.displayName = "Card";

export const CardHeader = React.forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("flex flex-col space-y-1.5 p-6", className)} {...props} />
));
CardHeader.displayName = "CardHeader";

export const CardTitle = React.forwardRef(({ className, ...props }, ref) => (
  <h3 ref={ref} className={cn("text-2xl font-semibold leading-none tracking-tight", className)} {...props} />
));
CardTitle.displayName = "CardTitle";

export const CardDescription = React.forwardRef(({ className, ...props }, ref) => (
  <p ref={ref} className={cn("text-sm text-muted-foreground", className)} {...props} />
));
CardDescription.displayName = "CardDescription";

export const CardContent = React.forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("p-6 pt-0", className)} {...props} />
));
CardContent.displayName = "CardContent";

export const CardFooter = React.forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("flex items-center p-6 pt-0", className)} {...props} />
));
CardFooter.displayName = "CardFooter";

// Progress component
export const Progress = React.forwardRef(({ className, value, max = 100, ...props }, ref) => {
  return (
    <div
      ref={ref}
      className={cn("relative h-2 w-full overflow-hidden rounded-full bg-secondary", className)}
      {...props}
    >
      <div
        className="h-full w-full flex-1 bg-primary transition-all"
        style={{ transform: `translateX(-${100 - (value / max) * 100}%)` }}
      />
    </div>
  );
});
Progress.displayName = "Progress";

// Tooltip component
export const Tooltip = ({ children, content }) => {
  const [isVisible, setIsVisible] = React.useState(false);

  return (
    <div className="relative inline-block">
      <div
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
        className="inline-flex items-center"
      >
        {children}
      </div>
      {isVisible && (
        <div className="absolute bottom-full left-1/2 z-50 mb-2 -translate-x-1/2 rounded-md bg-gray-800 px-3 py-2 text-xs text-white shadow-lg">
          {content}
          <div className="absolute -bottom-1 left-1/2 h-2 w-2 -translate-x-1/2 rotate-45 bg-gray-800"></div>
        </div>
      )}
    </div>
  );
};

// Step indicator component
export const StepIndicator = ({ steps, currentStep }) => {
  return (
    <div className="flex w-full items-center justify-between">
      {steps.map((step, index) => (
        <div key={index} className="flex flex-1 items-center">
          <div
            className={cn(
              "flex h-8 w-8 items-center justify-center rounded-full text-sm font-medium",
              index < currentStep
                ? "bg-green-100 text-green-700"
                : index === currentStep
                  ? "bg-blue-600 text-white"
                  : "bg-gray-100 text-gray-400",
            )}
          >
            {index < currentStep ? <CheckCircle2 className="h-4 w-4" /> : index + 1}
          </div>
          {index < steps.length - 1 && (
            <div
              className={cn(
                "h-1 flex-1",
                index < currentStep
                  ? "bg-green-400"
                  : index === currentStep
                    ? "bg-gradient-to-r from-blue-600 to-gray-200"
                    : "bg-gray-200",
              )}
            />
          )}
        </div>
      ))}
    </div>
  );
};

// Alert component
export const Alert = ({ variant = "info", title, message, onClose }) => {
  const getIcon = () => {
    switch (variant) {
      case "error":
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      case "warning":
        return <AlertTriangle className="h-4 w-4 text-amber-500" />;
      case "success":
        return <CheckCircle2 className="h-4 w-4 text-green-500" />;
      case "info":
      default:
        return <Info className="h-4 w-4 text-blue-500" />;
    }
  };

  const getStyles = () => {
    switch (variant) {
      case "error":
        return "border-red-200 bg-red-50 text-red-700";
      case "warning":
        return "border-amber-200 bg-amber-50 text-amber-700";
      case "success":
        return "border-green-200 bg-green-50 text-green-700";
      case "info":
      default:
        return "border-blue-200 bg-blue-50 text-blue-700";
    }
  };

  return (
    <div className={`rounded-lg border p-4 ${getStyles()}`}>
      <div className="flex items-start">
        <div className="flex-shrink-0">{getIcon()}</div>
        <div className="ml-3 flex-1">
          {title && <h3 className="text-sm font-medium">{title}</h3>}
          <div className="mt-1 text-sm">{message}</div>
        </div>
        {onClose && (
          <button
            type="button"
            className="ml-auto -mx-1.5 -my-1.5 inline-flex h-6 w-6 items-center justify-center rounded-md p-1 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-gray-400"
            onClick={onClose}
          >
            <span className="sr-only">Close</span>
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>
    </div>
  );
};
