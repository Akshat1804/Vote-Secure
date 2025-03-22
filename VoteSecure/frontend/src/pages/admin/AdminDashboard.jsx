"use client"

import React, { useState } from "react"
import { Link, Outlet, useLocation, useNavigate } from "react-router-dom"
import { ChevronDown, LogOut, Menu, Shield, User, Users, X } from "lucide-react"

// Simple utility function for combining class names without dependencies
const cn = (...classes) => {
  return classes
    .filter(Boolean)
    .join(" ")
    .replace(/border-border/g, "border")
}

// Badge component
const Badge = React.forwardRef(({ className, variant, ...props }, ref) => {
  return (
    <div
      ref={ref}
      className={cn(
        "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
        variant === "default" && "border-transparent bg-primary text-primary-foreground hover:bg-primary/80",
        variant === "secondary" && "border-transparent bg-secondary text-secondary-foreground hover:bg-secondary/80",
        variant === "success" && "border-transparent bg-green-100 text-green-800 hover:bg-green-200",
        variant === "destructive" &&
          "border-transparent bg-destructive text-destructive-foreground hover:bg-destructive/80",
        variant === "outline" && "text-foreground",
        variant === "warning" && "border-transparent bg-yellow-100 text-yellow-800 hover:bg-yellow-200",
        className,
      )}
      {...props}
    />
  )
})
Badge.displayName = "Badge"

// Button component
const Button = React.forwardRef(
  ({ className, variant = "default", size = "default", asChild = false, ...props }, ref) => {
    const Comp = asChild ? React.Fragment : "button"

    let variantClasses = ""
    if (variant === "default") variantClasses = "bg-primary text-primary-foreground hover:bg-primary/90"
    else if (variant === "destructive")
      variantClasses = "bg-destructive text-destructive-foreground hover:bg-destructive/90"
    else if (variant === "outline")
      variantClasses = "border border-input bg-background hover:bg-accent hover:text-accent-foreground"
    else if (variant === "secondary") variantClasses = "bg-secondary text-secondary-foreground hover:bg-secondary/80"
    else if (variant === "ghost") variantClasses = "hover:bg-accent hover:text-accent-foreground"
    else if (variant === "link") variantClasses = "text-primary underline-offset-4 hover:underline"

    let sizeClasses = ""
    if (size === "default") sizeClasses = "h-10 px-4 py-2"
    else if (size === "sm") sizeClasses = "h-9 rounded-md px-3"
    else if (size === "lg") sizeClasses = "h-11 rounded-md px-8"
    else if (size === "icon") sizeClasses = "h-10 w-10"

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
    )
  },
)
Button.displayName = "Button"

export default function AdminDashboard() {
  const navigate = useNavigate()
  const location = useLocation()
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [adminData, setAdminData] = useState({
    name: "Admin User",
    role: "Election Administrator",
    id: "ADMIN123456",
  })

  // Function to handle logout
  const handleLogout = () => {
    // Navigate to home
    navigate("/")
  }

  // Navigation items - simplified to focus only on voter registration
  const navigationItems = [
    {
      name: "Dashboard",
      path: "/admin",
      icon: Users,
    },
    {
      name: "Register New Voter",
      path: "/admin/register-user",
      icon: User,
    },
    {
      name: "Voter Management",
      path: "/admin/voters",
      icon: Users,
    },
  ]

  return (
    <div className="flex min-h-screen bg-gray-100">
      {/* Mobile Sidebar Overlay */}
      {sidebarOpen && (
        <div className="fixed inset-0 z-40 bg-black/50 md:hidden" onClick={() => setSidebarOpen(false)} />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-50 w-64 transform bg-white shadow-lg transition-transform duration-300 ease-in-out md:relative md:translate-x-0 ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex h-16 items-center justify-between border-b px-4">
          <Link to="/admin" className="flex items-center space-x-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-r from-blue-600 to-purple-600">
              <Shield className="h-4 w-4 text-white" />
            </div>
            <span className="text-lg font-bold text-gray-900">Admin Panel</span>
          </Link>
          <button
            className="rounded-md p-1 text-gray-500 hover:bg-gray-100 hover:text-gray-700 md:hidden"
            onClick={() => setSidebarOpen(false)}
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="flex flex-col py-4">
          <div className="px-4 py-2">
            <div className="flex items-center space-x-3 rounded-lg bg-gradient-to-r from-blue-50 to-purple-50 p-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-r from-blue-600 to-purple-600 text-white">
                {adminData.name.charAt(0)}
              </div>
              <div className="flex-1 overflow-hidden">
                <p className="truncate font-medium text-gray-900">{adminData.name}</p>
                <p className="truncate text-xs text-gray-500">{adminData.role}</p>
              </div>
            </div>
          </div>

          <nav className="mt-4 space-y-1 px-2">
            {navigationItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`flex w-full items-center space-x-3 rounded-md px-3 py-2 text-sm font-medium transition-colors ${
                  location.pathname === item.path
                    ? "bg-gradient-to-r from-blue-50 to-purple-50 text-blue-700"
                    : "text-gray-700 hover:bg-gray-100 hover:text-gray-900"
                }`}
              >
                <item.icon className="h-5 w-5" />
                <span>{item.name}</span>
                {item.badge && (
                  <Badge variant={item.badge.variant} className="ml-auto">
                    {item.badge.text}
                  </Badge>
                )}
              </Link>
            ))}
          </nav>

          <div className="mt-auto px-4 py-2">
            <button
              className="flex w-full items-center space-x-3 rounded-md px-3 py-2 text-sm font-medium text-red-600 transition-colors hover:bg-red-50"
              onClick={handleLogout}
            >
              <LogOut className="h-5 w-5" />
              <span>Logout</span>
            </button>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex flex-1 flex-col">
        {/* Header */}
        <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b bg-white px-4 md:px-6">
          <div className="flex items-center">
            <button
              className="mr-4 rounded-md p-1 text-gray-500 hover:bg-gray-100 hover:text-gray-700 md:hidden"
              onClick={() => setSidebarOpen(true)}
            >
              <Menu className="h-6 w-6" />
            </button>
            <h1 className="text-lg font-bold text-gray-900">
              {navigationItems.find((item) => item.path === location.pathname)?.name || "Admin Dashboard"}
            </h1>
          </div>

          <div className="flex items-center space-x-3">
            <div className="relative">
              <button className="flex items-center space-x-1 rounded-md p-1 text-gray-700 hover:bg-gray-100">
                <div className="h-8 w-8 rounded-full bg-gradient-to-r from-blue-600 to-purple-600 flex items-center justify-center text-white font-medium">
                  {adminData.name.charAt(0)}
                </div>
                <ChevronDown className="h-4 w-4" />
              </button>
            </div>
          </div>
        </header>

        {/* Main Content Area */}
        <main className="flex-1 overflow-auto">
          <Outlet />
        </main>
      </div>
    </div>
  )
}

