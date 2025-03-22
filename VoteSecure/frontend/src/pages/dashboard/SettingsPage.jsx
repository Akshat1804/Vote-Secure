"use client"

import React, { useState, useEffect } from "react"
import { AlertCircle, Bell, Check, Globe, Info, Lock, Mail, Phone, Save, Shield, User } from "lucide-react"
import { useVoters } from "../../context/VoterContext"

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

const CardFooter = React.forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("flex items-center p-6 pt-0", className)} {...props} />
))
CardFooter.displayName = "CardFooter"

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

// Switch component
const Switch = React.forwardRef(({ className, ...props }, ref) => (
  <button
    ref={ref}
    role="switch"
    className={cn(
      "peer inline-flex h-6 w-11 shrink-0 cursor-pointer items-center rounded-full border-2 border-transparent transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:cursor-not-allowed disabled:opacity-50 data-[state=checked]:bg-primary data-[state=unchecked]:bg-input",
      className,
    )}
    {...props}
  >
    <span
      className={cn(
        "pointer-events-none block h-5 w-5 rounded-full bg-background shadow-lg ring-0 transition-transform data-[state=checked]:translate-x-5 data-[state=unchecked]:translate-x-0",
      )}
    />
  </button>
))
Switch.displayName = "Switch"

export default function SettingsPage() {
  const { voters } = useVoters()
  const [currentVoter, setCurrentVoter] = useState(null)
  const [userData, setUserData] = useState({
    name: "John Doe",
    email: "j****@example.com",
    phone: "******7890",
    language: "English",
  })

  const [notificationSettings, setNotificationSettings] = useState({
    email: true,
    sms: true,
    app: true,
    electionReminders: true,
    documentUpdates: true,
    pollingStationChanges: true,
    securityAlerts: true,
  })

  const [securitySettings, setSecuritySettings] = useState({
    twoFactorAuth: false,
    loginNotifications: true,
    biometricLogin: false,
  })

  const [activeTab, setActiveTab] = useState("profile")
  const [showSaveSuccess, setShowSaveSuccess] = useState(false)

  // Get user data from session storage and find the corresponding voter
  useEffect(() => {
    const storedUserData = sessionStorage.getItem("userData")
    if (storedUserData) {
      const parsedUserData = JSON.parse(storedUserData)
      setUserData({
        ...userData,
        ...parsedUserData,
      })

      // Find the voter with matching voter ID or Aadhaar
      const matchingVoter = voters.find(
        (voter) => voter.voterID === parsedUserData.voterID || voter.aadhaar === parsedUserData.aadhaar,
      )

      if (matchingVoter) {
        setCurrentVoter(matchingVoter)
        // Update userData with voter data
        setUserData({
          name: matchingVoter.name || parsedUserData.name,
          email: matchingVoter.email || parsedUserData.email,
          phone: matchingVoter.phone || parsedUserData.phone,
          language: userData.language, // Keep the selected language
        })
      }
    }
  }, [voters])

  // Function to handle form submission
  const handleSubmit = (e) => {
    e.preventDefault()
    // Simulate saving
    setShowSaveSuccess(true)
    setTimeout(() => {
      setShowSaveSuccess(false)
    }, 3000)
  }

  // Function to toggle notification settings
  const toggleNotification = (setting) => {
    setNotificationSettings({
      ...notificationSettings,
      [setting]: !notificationSettings[setting],
    })
  }

  // Function to toggle security settings
  const toggleSecurity = (setting) => {
    setSecuritySettings({
      ...securitySettings,
      [setting]: !securitySettings[setting],
    })
  }

  return (
    <div className="p-4 md:p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
        <p className="text-gray-500">Manage your account preferences and settings</p>
      </div>

      {/* Settings Tabs */}
      <div className="mb-6 flex border-b">
        <button
          className={`border-b-2 px-4 py-2 text-sm font-medium ${
            activeTab === "profile"
              ? "border-blue-600 text-blue-600"
              : "border-transparent text-gray-500 hover:text-gray-700"
          }`}
          onClick={() => setActiveTab("profile")}
        >
          Profile
        </button>
        <button
          className={`border-b-2 px-4 py-2 text-sm font-medium ${
            activeTab === "notifications"
              ? "border-blue-600 text-blue-600"
              : "border-transparent text-gray-500 hover:text-gray-700"
          }`}
          onClick={() => setActiveTab("notifications")}
        >
          Notifications
        </button>
        <button
          className={`border-b-2 px-4 py-2 text-sm font-medium ${
            activeTab === "security"
              ? "border-blue-600 text-blue-600"
              : "border-transparent text-gray-500 hover:text-gray-700"
          }`}
          onClick={() => setActiveTab("security")}
        >
          Security
        </button>
        <button
          className={`border-b-2 px-4 py-2 text-sm font-medium ${
            activeTab === "language"
              ? "border-blue-600 text-blue-600"
              : "border-transparent text-gray-500 hover:text-gray-700"
          }`}
          onClick={() => setActiveTab("language")}
        >
          Language
        </button>
      </div>

      {/* Profile Settings */}
      {activeTab === "profile" && (
        <Card>
          <CardHeader>
            <CardTitle>Profile Information</CardTitle>
            <CardDescription>Update your personal information</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit}>
              <div className="space-y-4">
                <div className="flex flex-col items-center space-y-4 sm:flex-row sm:space-x-4 sm:space-y-0">
                  {currentVoter?.avatar ? (
                    <div className="flex h-24 w-24 items-center justify-center rounded-full bg-gradient-to-r from-blue-600 to-purple-600 p-1 overflow-hidden">
                      <img
                        src={currentVoter.avatar || "/placeholder.svg"}
                        alt={currentVoter.name}
                        className="h-full w-full rounded-full object-cover"
                        onError={(e) => {
                          e.target.onerror = null
                          e.target.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(currentVoter.name)}&background=random&color=fff&size=128`
                        }}
                      />
                    </div>
                  ) : (
                    <div className="flex h-24 w-24 items-center justify-center rounded-full bg-gradient-to-r from-blue-600 to-purple-600 p-1">
                      <div className="flex h-full w-full items-center justify-center rounded-full bg-white">
                        <User className="h-12 w-12 text-gray-400" />
                      </div>
                    </div>
                  )}
                  <div>
                    <h3 className="text-lg font-medium text-gray-900">{currentVoter?.name || userData.name}</h3>
                    <p className="text-sm text-gray-500">
                      Voter ID: {currentVoter?.id || currentVoter?.voterID || "ABC12345678"}
                    </p>
                    <div className="mt-2 rounded-md bg-yellow-50 px-3 py-1.5 text-xs text-yellow-800">
                      Profile details are synced with Aadhaar database
                    </div>
                  </div>
                </div>

                <div className="grid gap-4 pt-4 md:grid-cols-2">
                  <div>
                    <label htmlFor="name" className="mb-2 block text-sm font-medium text-gray-700">
                      Full Name
                    </label>
                    <input
                      id="name"
                      type="text"
                      value={currentVoter?.name || userData.name}
                      readOnly
                      className="w-full rounded-md border border-gray-200 bg-gray-50 p-2 text-sm"
                    />
                  </div>
                  <div>
                    <label htmlFor="email" className="mb-2 block text-sm font-medium text-gray-700">
                      Email Address
                    </label>
                    <input
                      id="email"
                      type="email"
                      value={currentVoter?.email || userData.email}
                      readOnly
                      className="w-full rounded-md border border-gray-200 bg-gray-50 p-2 text-sm"
                    />
                  </div>
                  <div>
                    <label htmlFor="phone" className="mb-2 block text-sm font-medium text-gray-700">
                      Phone Number
                    </label>
                    <input
                      id="phone"
                      type="tel"
                      value={currentVoter?.phone || userData.phone}
                      readOnly
                      className="w-full rounded-md border border-gray-200 bg-gray-50 p-2 text-sm"
                    />
                  </div>
                  <div>
                    <label htmlFor="address" className="mb-2 block text-sm font-medium text-gray-700">
                      Address
                    </label>
                    <textarea
                      id="address"
                      rows={3}
                      readOnly
                      className="w-full rounded-md border border-gray-200 bg-gray-50 p-2 text-sm"
                      defaultValue={currentVoter?.address || "123 Main Street, Bangalore, Karnataka"}
                    />
                  </div>
                </div>

                <div className="rounded-lg bg-blue-50 p-4">
                  <div className="flex items-start space-x-3">
                    <Info className="mt-0.5 h-5 w-5 text-blue-600" />
                    <div>
                      <h4 className="text-sm font-medium text-blue-800">Aadhaar Linked Profile</h4>
                      <p className="mt-1 text-sm text-blue-700">
                        Your profile information is linked to your Aadhaar card and cannot be modified directly. If you
                        need to update your details, please visit your nearest Aadhaar enrollment center.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </form>
          </CardContent>
        </Card>
      )}

      {/* Notification Settings */}
      {activeTab === "notifications" && (
        <Card>
          <CardHeader>
            <CardTitle>Notification Preferences</CardTitle>
            <CardDescription>Manage how you receive notifications</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              <div>
                <h3 className="mb-4 text-lg font-medium text-gray-900">Notification Channels</h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-100">
                        <Mail className="h-5 w-5 text-blue-600" />
                      </div>
                      <div>
                        <p className="font-medium text-gray-900">Email Notifications</p>
                        <p className="text-sm text-gray-500">Receive notifications via email</p>
                      </div>
                    </div>
                    <Switch
                      data-state={notificationSettings.email ? "checked" : "unchecked"}
                      onClick={() => toggleNotification("email")}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-full bg-green-100">
                        <Phone className="h-5 w-5 text-green-600" />
                      </div>
                      <div>
                        <p className="font-medium text-gray-900">SMS Notifications</p>
                        <p className="text-sm text-gray-500">Receive notifications via SMS</p>
                      </div>
                    </div>
                    <Switch
                      data-state={notificationSettings.sms ? "checked" : "unchecked"}
                      onClick={() => toggleNotification("sms")}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-full bg-purple-100">
                        <Bell className="h-5 w-5 text-purple-600" />
                      </div>
                      <div>
                        <p className="font-medium text-gray-900">App Notifications</p>
                        <p className="text-sm text-gray-500">Receive in-app notifications</p>
                      </div>
                    </div>
                    <Switch
                      data-state={notificationSettings.app ? "checked" : "unchecked"}
                      onClick={() => toggleNotification("app")}
                    />
                  </div>
                </div>
              </div>

              <div>
                <h3 className="mb-4 text-lg font-medium text-gray-900">Notification Types</h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-gray-900">Election Reminders</p>
                      <p className="text-sm text-gray-500">Notifications about upcoming elections</p>
                    </div>
                    <Switch
                      data-state={notificationSettings.electionReminders ? "checked" : "unchecked"}
                      onClick={() => toggleNotification("electionReminders")}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-gray-900">Document Updates</p>
                      <p className="text-sm text-gray-500">Notifications about document verification status</p>
                    </div>
                    <Switch
                      data-state={notificationSettings.documentUpdates ? "checked" : "unchecked"}
                      onClick={() => toggleNotification("documentUpdates")}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-gray-900">Polling Station Changes</p>
                      <p className="text-sm text-gray-500">Notifications about changes to your polling station</p>
                    </div>
                    <Switch
                      data-state={notificationSettings.pollingStationChanges ? "checked" : "unchecked"}
                      onClick={() => toggleNotification("pollingStationChanges")}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-gray-900">Security Alerts</p>
                      <p className="text-sm text-gray-500">Notifications about security-related activities</p>
                    </div>
                    <Switch
                      data-state={notificationSettings.securityAlerts ? "checked" : "unchecked"}
                      onClick={() => toggleNotification("securityAlerts")}
                    />
                  </div>
                </div>
              </div>

              <div className="rounded-lg bg-yellow-50 p-4">
                <div className="flex items-start space-x-3">
                  <AlertCircle className="mt-0.5 h-5 w-5 text-yellow-600" />
                  <div>
                    <h4 className="text-sm font-medium text-yellow-800">Important Notice</h4>
                    <p className="mt-1 text-sm text-yellow-700">
                      Election day reminders cannot be disabled as they are mandatory notifications required by the
                      Election Commission.
                    </p>
                  </div>
                </div>
              </div>

              <div className="flex justify-end">
                <Button onClick={handleSubmit}>
                  <Save className="mr-2 h-4 w-4" />
                  Save Preferences
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Security Settings */}
      {activeTab === "security" && (
        <Card>
          <CardHeader>
            <CardTitle>Security Settings</CardTitle>
            <CardDescription>Manage your account security</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              <div>
                <h3 className="mb-4 text-lg font-medium text-gray-900">Account Security</h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-100">
                        <Lock className="h-5 w-5 text-blue-600" />
                      </div>
                      <div>
                        <p className="font-medium text-gray-900">Change Password</p>
                        <p className="text-sm text-gray-500">Update your account password</p>
                      </div>
                    </div>
                    <Button variant="outline" size="sm">
                      Change
                    </Button>
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-full bg-purple-100">
                        <Shield className="h-5 w-5 text-purple-600" />
                      </div>
                      <div>
                        <p className="font-medium text-gray-900">Two-Factor Authentication</p>
                        <p className="text-sm text-gray-500">Add an extra layer of security to your account</p>
                      </div>
                    </div>
                    <Switch
                      data-state={securitySettings.twoFactorAuth ? "checked" : "unchecked"}
                      onClick={() => toggleSecurity("twoFactorAuth")}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-full bg-green-100">
                        <Bell className="h-5 w-5 text-green-600" />
                      </div>
                      <div>
                        <p className="font-medium text-gray-900">Login Notifications</p>
                        <p className="text-sm text-gray-500">Get notified when someone logs into your account</p>
                      </div>
                    </div>
                    <Switch
                      data-state={securitySettings.loginNotifications ? "checked" : "unchecked"}
                      onClick={() => toggleSecurity("loginNotifications")}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-full bg-yellow-100">
                        <User className="h-5 w-5 text-yellow-600" />
                      </div>
                      <div>
                        <p className="font-medium text-gray-900">Biometric Login</p>
                        <p className="text-sm text-gray-500">Use fingerprint or face recognition to login</p>
                      </div>
                    </div>
                    <Switch
                      data-state={securitySettings.biometricLogin ? "checked" : "unchecked"}
                      onClick={() => toggleSecurity("biometricLogin")}
                    />
                  </div>
                </div>
              </div>

              <div className="rounded-lg bg-blue-50 p-4">
                <div className="flex items-start space-x-3">
                  <Info className="mt-0.5 h-5 w-5 text-blue-600" />
                  <div>
                    <h4 className="text-sm font-medium text-blue-800">Security Recommendation</h4>
                    <p className="mt-1 text-sm text-blue-700">
                      We recommend enabling two-factor authentication for enhanced security. This adds an extra layer of
                      protection to your account by requiring a verification code in addition to your password.
                    </p>
                  </div>
                </div>
              </div>

              <div className="flex justify-end">
                <Button onClick={handleSubmit}>
                  <Save className="mr-2 h-4 w-4" />
                  Save Settings
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Language Settings */}
      {activeTab === "language" && (
        <Card>
          <CardHeader>
            <CardTitle>Language Preferences</CardTitle>
            <CardDescription>Choose your preferred language</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              <div>
                <h3 className="mb-4 text-lg font-medium text-gray-900">Display Language</h3>
                <div className="space-y-4">
                  <div className="flex items-center space-x-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-100">
                      <Globe className="h-5 w-5 text-blue-600" />
                    </div>
                    <div>
                      <p className="font-medium text-gray-900">Application Language</p>
                      <p className="text-sm text-gray-500">Select the language for the application interface</p>
                    </div>
                  </div>

                  <div className="grid gap-4 md:grid-cols-2">
                    {["English", "Hindi", "Gujarati", "Tamil", "Bengali"].map((language) => (
                      <div
                        key={language}
                        className={`flex cursor-pointer items-center justify-between rounded-lg border p-3 ${
                          userData.language === language
                            ? "border-blue-500 bg-blue-50"
                            : "border-gray-200 hover:border-blue-200 hover:bg-blue-50"
                        }`}
                        onClick={() => setUserData({ ...userData, language })}
                      >
                        <span className="font-medium">{language}</span>
                        {userData.language === language && (
                          <div className="flex h-5 w-5 items-center justify-center rounded-full bg-blue-500 text-white">
                            <Check className="h-3 w-3" />
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="rounded-lg bg-blue-50 p-4">
                <div className="flex items-start space-x-3">
                  <Info className="mt-0.5 h-5 w-5 text-blue-600" />
                  <div>
                    <h4 className="text-sm font-medium text-blue-800">Language Support</h4>
                    <p className="mt-1 text-sm text-blue-700">
                      The application supports multiple languages to make it accessible to a wider audience. If you
                      don't see your preferred language, more languages will be added in future updates.
                    </p>
                  </div>
                </div>
              </div>

              <div className="flex justify-end">
                <Button onClick={handleSubmit}>
                  <Save className="mr-2 h-4 w-4" />
                  Save Preferences
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Success Message */}
      {showSaveSuccess && (
        <div className="fixed bottom-4 right-4 z-50 rounded-lg bg-green-100 p-4 shadow-lg">
          <div className="flex items-center space-x-2">
            <div className="flex h-6 w-6 items-center justify-center rounded-full bg-green-500 text-white">
              <Check className="h-4 w-4" />
            </div>
            <p className="font-medium text-green-800">Settings saved successfully!</p>
          </div>
        </div>
      )}
    </div>
  )
}

