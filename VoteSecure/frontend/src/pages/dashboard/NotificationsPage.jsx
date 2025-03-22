"use client"

import React, { useState } from "react"
import {
  AlertCircle,
  Calendar,
  Check,
  ChevronRight,
  Clock,
  FileText,
  Filter,
  MapPin,
  Search,
  Shield,
  Trash,
} from "lucide-react"

// Simple utility function for combining class names without dependencies
const cn = (...classes) => {
  return classes
    .filter(Boolean)
    .join(" ")
    .replace(/border-border/g, "border")
}

// Card components
const Card = React.forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("rounded-lg border bg-card text-card-foreground shadow-sm", className)} {...props} />
))
Card.displayName = "Card"

const CardHeader = React.forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("flex flex-col space-y-1.5 p-6", className)} {...props} />
))
CardHeader.displayName = "CardHeader"

const CardTitle = React.forwardRef(({ className, ...props }, ref) => (
  <h3 ref={ref} className={cn("text-2xl font-semibold leading-none tracking-tight", className)} {...props} />
))
CardTitle.displayName = "CardTitle"

const CardDescription = React.forwardRef(({ className, ...props }, ref) => (
  <p ref={ref} className={cn("text-sm text-muted-foreground", className)} {...props} />
))
CardDescription.displayName = "CardDescription"

const CardContent = React.forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("p-6 pt-0", className)} {...props} />
))
CardContent.displayName = "CardContent"

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

export default function NotificationsPage() {
  const [notifications, setNotifications] = useState([
    {
      id: 1,
      title: "Election Reminder",
      message: "General Elections 2025 are scheduled for April 15th.",
      time: "2 hours ago",
      date: "March 18, 2025",
      icon: Calendar,
      color: "bg-blue-100 text-blue-600",
      category: "Election",
      read: false,
    },
    {
      id: 2,
      title: "Document Verification",
      message: "Your voter ID has been successfully verified.",
      time: "1 day ago",
      date: "March 17, 2025",
      icon: FileText,
      color: "bg-green-100 text-green-600",
      category: "Document",
      read: false,
    },
    {
      id: 3,
      title: "New Polling Station",
      message: "Your polling station has been updated. Please check details.",
      time: "3 days ago",
      date: "March 15, 2025",
      icon: MapPin,
      color: "bg-yellow-100 text-yellow-600",
      category: "Polling",
      read: true,
    },
    {
      id: 4,
      title: "Security Alert",
      message: "A new device was used to access your account. If this wasn't you, please contact support.",
      time: "1 week ago",
      date: "March 11, 2025",
      icon: Shield,
      color: "bg-red-100 text-red-600",
      category: "Security",
      read: true,
    },
    {
      id: 5,
      title: "Voting Hours Update",
      message: "Voting hours for the upcoming election have been extended to 8:00 AM - 7:00 PM.",
      time: "2 weeks ago",
      date: "March 4, 2025",
      icon: Clock,
      color: "bg-purple-100 text-purple-600",
      category: "Election",
      read: true,
    },
  ])

  const [searchQuery, setSearchQuery] = useState("")
  const [filterCategory, setFilterCategory] = useState("all")
  const [showReadNotifications, setShowReadNotifications] = useState(true)
  const [selectedNotification, setSelectedNotification] = useState(null)

  // Function to mark notification as read
  const markAsRead = (id) => {
    setNotifications(
      notifications.map((notification) => (notification.id === id ? { ...notification, read: true } : notification)),
    )
  }

  // Function to mark all notifications as read
  const markAllAsRead = () => {
    setNotifications(
      notifications.map((notification) => ({
        ...notification,
        read: true,
      })),
    )
  }

  // Function to delete notification
  const deleteNotification = (id) => {
    setNotifications(notifications.filter((notification) => notification.id !== id))
    if (selectedNotification?.id === id) {
      setSelectedNotification(null)
    }
  }

  // Function to clear all notifications
  const clearAllNotifications = () => {
    setNotifications([])
    setSelectedNotification(null)
  }

  // Filter notifications based on search query, category, and read status
  const filteredNotifications = notifications.filter((notification) => {
    const matchesSearch =
      notification.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      notification.message.toLowerCase().includes(searchQuery.toLowerCase())

    const matchesCategory =
      filterCategory === "all" || notification.category.toLowerCase() === filterCategory.toLowerCase()

    const matchesReadStatus = showReadNotifications || !notification.read

    return matchesSearch && matchesCategory && matchesReadStatus
  })

  // Count unread notifications
  const unreadCount = notifications.filter((notification) => !notification.read).length

  return (
    <div className="p-4 md:p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Notifications</h1>
        <p className="text-gray-500">Stay updated with important information</p>
      </div>

      {/* Notification Actions */}
      <div className="mb-6 flex flex-wrap items-center gap-4">
        <div className="relative w-full sm:max-w-xs">
          <Search className="absolute left-3 top-2.5 h-4 w-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search notifications..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full rounded-md border border-gray-200 bg-white py-2 pl-9 pr-3 text-sm placeholder:text-gray-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        </div>

        <div className="flex items-center space-x-2">
          <Filter className="h-4 w-4 text-gray-500" />
          <span className="text-sm text-gray-500">Filter:</span>
          <select
            value={filterCategory}
            onChange={(e) => setFilterCategory(e.target.value)}
            className="rounded-md border border-gray-200 bg-white px-3 py-1.5 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          >
            <option value="all">All Categories</option>
            <option value="election">Election</option>
            <option value="document">Document</option>
            <option value="polling">Polling</option>
            <option value="security">Security</option>
          </select>
        </div>

        <div className="flex items-center space-x-2">
          <input
            type="checkbox"
            id="show-read"
            checked={showReadNotifications}
            onChange={() => setShowReadNotifications(!showReadNotifications)}
            className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
          />
          <label htmlFor="show-read" className="text-sm text-gray-700">
            Show read notifications
          </label>
        </div>

        <div className="ml-auto flex space-x-2">
          <Button variant="outline" size="sm" onClick={markAllAsRead} disabled={unreadCount === 0}>
            <Check className="mr-1.5 h-4 w-4" />
            Mark all as read
          </Button>
          <Button variant="destructive" size="sm" onClick={clearAllNotifications} disabled={notifications.length === 0}>
            <Trash className="mr-1.5 h-4 w-4" />
            Clear all
          </Button>
        </div>
      </div>

      {/* Notifications List */}
      <div className="grid gap-6 md:grid-cols-3">
        <div className="md:col-span-1">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Notifications</CardTitle>
                {unreadCount > 0 && (
                  <Badge variant="default" className="ml-2">
                    {unreadCount} New
                  </Badge>
                )}
              </div>
              <CardDescription>
                {notifications.length > 0
                  ? `You have ${notifications.length} notification${notifications.length !== 1 ? "s" : ""}`
                  : "No notifications"}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {filteredNotifications.length > 0 ? (
                  filteredNotifications.map((notification) => (
                    <div
                      key={notification.id}
                      className={`cursor-pointer rounded-lg border p-3 transition-all hover:border-blue-200 hover:bg-blue-50 ${
                        selectedNotification?.id === notification.id
                          ? "border-blue-500 bg-blue-50"
                          : notification.read
                            ? "border-gray-200"
                            : "border-blue-200 bg-blue-50/50"
                      }`}
                      onClick={() => {
                        setSelectedNotification(notification)
                        if (!notification.read) {
                          markAsRead(notification.id)
                        }
                      }}
                    >
                      <div className="flex items-start space-x-3">
                        <div
                          className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${notification.color}`}
                        >
                          <notification.icon className="h-4 w-4" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center justify-between">
                            <h4 className="font-medium text-gray-900 truncate">{notification.title}</h4>
                            {!notification.read && <span className="ml-2 h-2 w-2 rounded-full bg-blue-600"></span>}
                          </div>
                          <p className="mt-1 line-clamp-2 text-xs text-gray-600">{notification.message}</p>
                          <p className="mt-1 text-xs text-gray-400">{notification.time}</p>
                        </div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="rounded-lg border border-dashed p-6 text-center">
                    <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-gray-100">
                      <AlertCircle className="h-6 w-6 text-gray-400" />
                    </div>
                    <h3 className="text-sm font-medium text-gray-900">No notifications found</h3>
                    <p className="mt-2 text-xs text-gray-500">
                      {searchQuery || filterCategory !== "all" || !showReadNotifications
                        ? "Try adjusting your filters"
                        : "You don't have any notifications yet"}
                    </p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Notification Detail */}
        <div className="md:col-span-2">
          {selectedNotification ? (
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <div
                      className={`mr-3 flex h-10 w-10 items-center justify-center rounded-full ${selectedNotification.color}`}
                    >
                      <selectedNotification.icon className="h-5 w-5" />
                    </div>
                    <div>
                      <CardTitle>{selectedNotification.title}</CardTitle>
                      <CardDescription>{selectedNotification.date}</CardDescription>
                    </div>
                  </div>
                  <div className="flex space-x-2">
                    {!selectedNotification.read && (
                      <Button variant="outline" size="sm" onClick={() => markAsRead(selectedNotification.id)}>
                        <Check className="mr-1.5 h-4 w-4" />
                        Mark as read
                      </Button>
                    )}
                    <Button variant="destructive" size="sm" onClick={() => deleteNotification(selectedNotification.id)}>
                      <Trash className="mr-1.5 h-4 w-4" />
                      Delete
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="rounded-lg bg-gray-50 p-4">
                    <p className="text-gray-700">{selectedNotification.message}</p>
                  </div>

                  {selectedNotification.category === "Election" && (
                    <div className="rounded-lg border p-4">
                      <h4 className="mb-2 font-medium text-gray-900">Election Details</h4>
                      <div className="space-y-2">
                        <div className="flex items-center text-sm">
                          <Calendar className="mr-2 h-4 w-4 text-gray-400" />
                          <span>Date: April 15, 2025</span>
                        </div>
                        <div className="flex items-center text-sm">
                          <Clock className="mr-2 h-4 w-4 text-gray-400" />
                          <span>Time: 8:00 AM - 6:00 PM</span>
                        </div>
                        <div className="flex items-center text-sm">
                          <MapPin className="mr-2 h-4 w-4 text-gray-400" />
                          <span>Location: Your assigned polling station</span>
                        </div>
                      </div>
                      <Button className="mt-4" size="sm">
                        View Election Details
                        <ChevronRight className="ml-1 h-4 w-4" />
                      </Button>
                    </div>
                  )}

                  {selectedNotification.category === "Polling" && (
                    <div className="rounded-lg border p-4">
                      <h4 className="mb-2 font-medium text-gray-900">Polling Station Update</h4>
                      <div className="space-y-2">
                        <div className="flex items-center text-sm">
                          <MapPin className="mr-2 h-4 w-4 text-gray-400" />
                          <span>New Location: Government High School, Sector 12</span>
                        </div>
                        <div className="flex items-center text-sm">
                          <Clock className="mr-2 h-4 w-4 text-gray-400" />
                          <span>Booth Number: 42</span>
                        </div>
                      </div>
                      <Button className="mt-4" size="sm">
                        View on Map
                        <ChevronRight className="ml-1 h-4 w-4" />
                      </Button>
                    </div>
                  )}

                  {selectedNotification.category === "Document" && (
                    <div className="rounded-lg border p-4">
                      <h4 className="mb-2 font-medium text-gray-900">Document Status</h4>
                      <div className="space-y-2">
                        <div className="flex items-center text-sm">
                          <FileText className="mr-2 h-4 w-4 text-gray-400" />
                          <span>Document: Voter ID Card</span>
                        </div>
                        <div className="flex items-center text-sm">
                          <Check className="mr-2 h-4 w-4 text-green-500" />
                          <span className="text-green-600">Status: Verified</span>
                        </div>
                      </div>
                      <Button className="mt-4" size="sm">
                        View Documents
                        <ChevronRight className="ml-1 h-4 w-4" />
                      </Button>
                    </div>
                  )}

                  {selectedNotification.category === "Security" && (
                    <div className="rounded-lg border-2 border-red-200 bg-red-50 p-4">
                      <h4 className="mb-2 font-medium text-red-800">Security Alert</h4>
                      <div className="space-y-2">
                        <div className="flex items-start text-sm">
                          <AlertCircle className="mr-2 mt-0.5 h-4 w-4 text-red-500" />
                          <span className="text-red-700">
                            A new device was used to access your account on March 11, 2025 at 10:15 AM. If this wasn't
                            you, please change your password immediately and contact support.
                          </span>
                        </div>
                      </div>
                      <div className="mt-4 flex space-x-2">
                        <Button variant="destructive" size="sm">
                          Change Password
                        </Button>
                        <Button variant="outline" size="sm">
                          Contact Support
                        </Button>
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent className="p-6">
                <div className="flex flex-col items-center justify-center py-8 text-center">
                  <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-gray-100">
                    <AlertCircle className="h-8 w-8 text-gray-400" />
                  </div>
                  <h3 className="text-lg font-medium text-gray-900">No notification selected</h3>
                  <p className="mt-2 text-gray-500">Select a notification from the list to view its details</p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}

