"use client"

import React, { useState, useEffect } from "react"
import { Outlet, useNavigate, useLocation, Link } from "react-router-dom"
import {
  Calendar,
  ChevronDown,
  FileText,
  HelpCircle,
  Home,
  LogOut,
  MapPin,
  Menu,
  Settings,
  User,
  X,
  Vote,
} from "lucide-react"
import { useVoters } from "../context/VoterContext"

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

export default function DashboardLayout() {
  const navigate = useNavigate()
  const location = useLocation()
  const [userData, setUserData] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [currentLanguage, setCurrentLanguage] = useState("English")
  const [showLanguageSelector, setShowLanguageSelector] = useState(false)
  const { voters } = useVoters()
  const [currentVoter, setCurrentVoter] = useState(null)
  const [activeElection, setActiveElection] = useState(true) // Set to true if there's an active election

  // Get user data from session storage and find the corresponding voter
  useEffect(() => {
    const storedUserData = sessionStorage.getItem("userData")

    if (!storedUserData) {
      // If no user data in session, redirect back to home
      navigate("/")
      return
    }

    try {
      const parsedUserData = JSON.parse(storedUserData)
      setUserData(parsedUserData)

      // Find the voter with matching voter ID or Aadhaar
      const matchingVoter = voters.find(
        (voter) =>
          (voter.voterID && parsedUserData.voterID && voter.voterID === parsedUserData.voterID) ||
          (voter.aadhaar &&
            parsedUserData.aadhaar &&
            voter.aadhaar.replace(/\s/g, "") === parsedUserData.aadhaar.replace(/\s/g, "")),
      )

      if (matchingVoter) {
        // Create a complete voter object by merging matching voter with userData
        const completeVoter = {
          ...parsedUserData,
          ...matchingVoter,
          // Ensure we have the correct ID format
          id: matchingVoter.id || parsedUserData.voterID,
          voterID: matchingVoter.voterID || parsedUserData.voterID || matchingVoter.id,
          // Use the avatar from the matching voter if available, otherwise use from userData
          avatar: matchingVoter.avatar || parsedUserData.avatar,
        }
        setCurrentVoter(completeVoter)
      } else {
        // If no matching voter in voters array, use the userData as the voter data
        setCurrentVoter(parsedUserData)
      }
    } catch (error) {
      console.error("Error parsing user data:", error)
      // Handle corrupted data by redirecting to login
      navigate("/")
    }
  }, [navigate, voters])

  // Function to handle logout
  const handleLogout = () => {
    // Clear session storage
    sessionStorage.removeItem("aadhaarNumber")
    sessionStorage.removeItem("maskedPhone")
    sessionStorage.removeItem("userData")

    // Navigate to home
    navigate("/")
  }

  // Function to toggle language selector
  const toggleLanguageSelector = () => {
    setShowLanguageSelector(!showLanguageSelector)
  }

  // Function to change language
  const changeLanguage = (language) => {
    setCurrentLanguage(language)
    setShowLanguageSelector(false)
  }

  // Navigation items
  const navigationItems = [
    {
      name: "Dashboard",
      path: "/dashboard",
      icon: Home,
    },
    {
      name: "Profile",
      path: "/dashboard/profile",
      icon: User,
    },
    // NEW ITEM: Vote Now - only shown when there's an active election
    ...(activeElection
      ? [
          {
            name: "Vote Now",
            path: "/voting/verify",
            icon: Vote,
            badge: { text: "Active", variant: "success" },
          },
        ]
      : []),
    {
      name: "Elections",
      path: "/dashboard/elections",
      icon: Calendar,
      badge: { text: "New", variant: "success" },
    },
    {
      name: "Documents",
      path: "/dashboard/documents",
      icon: FileText,
    },
    {
      name: "Polling Map",
      path: "/dashboard/polling-map",
      icon: MapPin,
    },
    {
      name: "Settings",
      path: "/dashboard/settings",
      icon: Settings,
    },
    {
      name: "Help & Support",
      path: "/dashboard/help",
      icon: HelpCircle,
    },
  ]

  if (!userData) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-blue-50 via-white to-purple-50">
        <div className="text-center">
          <div className="mb-4 h-12 w-12 animate-spin rounded-full border-4 border-blue-200 border-t-blue-600 mx-auto"></div>
          <p className="text-gray-600">Loading user data...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex min-h-screen bg-gray-50">
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
          <Link to="/dashboard" className="flex items-center space-x-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-r from-blue-600 to-purple-600">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="h-4 w-4 text-white"
              >
                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                <polyline points="15 3 21 3 21 9" />
                <line x1="10" y1="14" x2="21" y2="3" />
              </svg>
            </div>
            <span className="text-lg font-bold text-gray-900">VoteSecure</span>
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
              {currentVoter?.avatar ? (
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-r from-blue-600 to-purple-600 text-white overflow-hidden">
                  <img
                    src={currentVoter.avatar || "/placeholder.svg"}
                    alt={currentVoter.name}
                    className="h-full w-full object-cover"
                    onError={(e) => {
                      e.target.onerror = null
                      e.target.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(currentVoter.name)}&background=random&color=fff&size=128`
                    }}
                  />
                </div>
              ) : (
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-r from-blue-600 to-purple-600 text-white">
                  {(currentVoter?.name || userData.name).charAt(0)}
                </div>
              )}
              <div className="flex-1 overflow-hidden">
                <p className="truncate font-medium text-gray-900">{currentVoter?.name || userData.name}</p>
                <p className="truncate text-xs text-gray-500">
                  Voter ID: {currentVoter?.id || currentVoter?.voterID || userData.voterID}
                </p>
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
            <div className="flex items-center justify-between mb-2">
              <button
                className="flex items-center space-x-2 rounded-md px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-100"
                onClick={toggleLanguageSelector}
              >
                <span>{currentLanguage}</span>
                <ChevronDown className="h-4 w-4" />
              </button>

              {showLanguageSelector && (
                <div className="absolute bottom-16 left-4 w-40 rounded-md border border-gray-200 bg-white shadow-lg">
                  {["English", "Hindi", "Gujarati", "Tamil", "Bengali"].map((language) => (
                    <button
                      key={language}
                      className={`block w-full px-4 py-2 text-left text-sm ${
                        currentLanguage === language ? "bg-blue-50 text-blue-700" : "text-gray-700 hover:bg-gray-50"
                      }`}
                      onClick={() => changeLanguage(language)}
                    >
                      {language}
                    </button>
                  ))}
                </div>
              )}
            </div>

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
              {navigationItems.find((item) => item.path === location.pathname)?.name || "Dashboard"}
            </h1>
          </div>

          <div className="flex items-center space-x-3">
            {/* Vote Now Button in Header - NEW ADDITION */}
            {activeElection && (
              <Link to="/voting/verify">
                <Button size="sm" className="bg-green-600 hover:bg-green-700 hidden md:flex">
                  Vote Now
                </Button>
              </Link>
            )}

            <div className="relative">
              <button className="flex items-center space-x-1 rounded-md p-1 text-gray-700 hover:bg-gray-100">
                {currentVoter?.avatar ? (
                  <div className="h-8 w-8 rounded-full overflow-hidden">
                    <img
                      src={currentVoter.avatar || "/placeholder.svg"}
                      alt={currentVoter.name}
                      className="h-full w-full object-cover"
                      onError={(e) => {
                        e.target.onerror = null
                        e.target.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(currentVoter.name)}&background=random&color=fff&size=128`
                      }}
                    />
                  </div>
                ) : (
                  <div className="h-8 w-8 rounded-full bg-gradient-to-r from-blue-600 to-purple-600 flex items-center justify-center text-white font-medium">
                    {(currentVoter?.name || userData.name).charAt(0)}
                  </div>
                )}
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

