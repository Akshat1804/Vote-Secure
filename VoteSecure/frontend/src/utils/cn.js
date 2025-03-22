// Simple utility function for combining class names without dependencies
export const cn = (...classes) => {
    return classes
      .filter(Boolean)
      .join(" ")
      .replace(/border-border/g, "border")
  }
  
  