"use client"

import React, { useState } from "react"
import {
  AlertCircle,
  ArrowRight,
  Book,
  ChevronDown,
  ChevronRight,
  FileText,
  HelpCircle,
  Mail,
  Phone,
  Search,
  Send,
  User,
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

export default function HelpSupportPage() {
  const [searchQuery, setSearchQuery] = useState("")
  const [activeCategory, setActiveCategory] = useState("all")
  const [activeFaq, setActiveFaq] = useState(null)
  const [chatMessages, setChatMessages] = useState([
    {
      id: 1,
      sender: "system",
      message: "Hello! How can I help you today?",
      time: "10:30 AM",
    },
  ])
  const [newMessage, setNewMessage] = useState("")

  // Mock FAQs data
  const faqs = [
    {
      id: 1,
      question: "How do I update my address information?",
      answer:
        "You can update your address information by going to the Profile section in your dashboard. Click on the Edit button next to your address, enter your new address details, and click Save. You will need to upload a new address proof document for verification.",
      category: "profile",
    },
    {
      id: 2,
      question: "What documents are required for voter registration?",
      answer:
        "For voter registration, you need to provide your Aadhaar card for identity verification and an address proof document such as a utility bill, rent agreement, or passport. All documents should be clear, legible, and in PDF, JPG, or PNG format.",
      category: "documents",
    },
    {
      id: 3,
      question: "How do I find my polling station?",
      answer:
        "You can find your assigned polling station by visiting the Polling Map section in your dashboard. The map will show your assigned polling station along with directions. You can also download the details for offline reference.",
      category: "voting",
    },
    {
      id: 4,
      question: "What should I do if I can't vote on election day?",
      answer:
        "If you cannot vote on election day, you may be eligible for postal voting or early voting options depending on your circumstances. Please contact your local election office or use the Help & Support section to get specific guidance for your situation.",
      category: "voting",
    },
    {
      id: 5,
      question: "How do I reset my password?",
      answer:
        "To reset your password, click on the 'Forgot Password' link on the login page. Enter your registered email address or mobile number to receive a password reset link or OTP. Follow the instructions to create a new password.",
      category: "security",
    },
    {
      id: 6,
      question: "What is the QR code used for?",
      answer:
        "The QR code is a secure digital identifier that is used for verification at the polling station. It contains encrypted information about your voter ID and is scanned at the polling booth to verify your identity before you cast your vote.",
      category: "voting",
    },
  ]

  // Filter FAQs based on search query and active category
  const filteredFaqs = faqs.filter((faq) => {
    const matchesSearch =
      faq.question.toLowerCase().includes(searchQuery.toLowerCase()) ||
      faq.answer.toLowerCase().includes(searchQuery.toLowerCase())

    const matchesCategory = activeCategory === "all" || faq.category === activeCategory

    return matchesSearch && matchesCategory
  })

  // Function to send a new chat message
  const sendMessage = () => {
    if (newMessage.trim() === "") return

    // Add user message
    const userMessage = {
      id: chatMessages.length + 1,
      sender: "user",
      message: newMessage,
      time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    }
    setChatMessages([...chatMessages, userMessage])
    setNewMessage("")

    // Simulate agent response after a short delay
    setTimeout(() => {
      const agentMessage = {
        id: chatMessages.length + 2,
        sender: "system",
        message:
          "Thank you for your message. Our support team will get back to you shortly. In the meantime, you might find an answer to your question in our FAQ section.",
        time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      }
      setChatMessages((prevMessages) => [...prevMessages, agentMessage])
    }, 1000)
  }

  return (
    <div className="p-4 md:p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Help & Support</h1>
        <p className="text-gray-500">Find answers to your questions and get assistance</p>
      </div>

      {/* Search Bar */}
      <div className="mb-6 relative">
        <Search className="absolute left-3 top-3 h-5 w-5 text-gray-400" />
        <input
          type="text"
          placeholder="Search for help topics..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full rounded-lg border border-gray-200 bg-white py-3 pl-10 pr-4 text-sm placeholder:text-gray-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
        />
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        {/* Help Categories */}
        <div className="md:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle>Help Categories</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <button
                  className={`flex w-full items-center justify-between rounded-md px-3 py-2 text-left text-sm font-medium ${
                    activeCategory === "all" ? "bg-blue-50 text-blue-700" : "text-gray-700 hover:bg-gray-100"
                  }`}
                  onClick={() => setActiveCategory("all")}
                >
                  <div className="flex items-center">
                    <HelpCircle className="mr-2 h-5 w-5" />
                    <span>All Topics</span>
                  </div>
                  <ChevronRight className="h-4 w-4" />
                </button>

                <button
                  className={`flex w-full items-center justify-between rounded-md px-3 py-2 text-left text-sm font-medium ${
                    activeCategory === "profile" ? "bg-blue-50 text-blue-700" : "text-gray-700 hover:bg-gray-100"
                  }`}
                  onClick={() => setActiveCategory("profile")}
                >
                  <div className="flex items-center">
                    <User className="mr-2 h-5 w-5" />
                    <span>Profile & Account</span>
                  </div>
                  <ChevronRight className="h-4 w-4" />
                </button>

                <button
                  className={`flex w-full items-center justify-between rounded-md px-3 py-2 text-left text-sm font-medium ${
                    activeCategory === "documents" ? "bg-blue-50 text-blue-700" : "text-gray-700 hover:bg-gray-100"
                  }`}
                  onClick={() => setActiveCategory("documents")}
                >
                  <div className="flex items-center">
                    <FileText className="mr-2 h-5 w-5" />
                    <span>Documents</span>
                  </div>
                  <ChevronRight className="h-4 w-4" />
                </button>

                <button
                  className={`flex w-full items-center justify-between rounded-md px-3 py-2 text-left text-sm font-medium ${
                    activeCategory === "voting" ? "bg-blue-50 text-blue-700" : "text-gray-700 hover:bg-gray-100"
                  }`}
                  onClick={() => setActiveCategory("voting")}
                >
                  <div className="flex items-center">
                    <AlertCircle className="mr-2 h-5 w-5" />
                    <span>Voting & Elections</span>
                  </div>
                  <ChevronRight className="h-4 w-4" />
                </button>

                <button
                  className={`flex w-full items-center justify-between rounded-md px-3 py-2 text-left text-sm font-medium ${
                    activeCategory === "security" ? "bg-blue-50 text-blue-700" : "text-gray-700 hover:bg-gray-100"
                  }`}
                  onClick={() => setActiveCategory("security")}
                >
                  <div className="flex items-center">
                    <AlertCircle className="mr-2 h-5 w-5" />
                    <span>Security & Privacy</span>
                  </div>
                  <ChevronRight className="h-4 w-4" />
                </button>
              </div>

              <div className="mt-6 rounded-lg bg-gradient-to-r from-blue-600 to-purple-600 p-4 text-white">
                <h3 className="font-medium">Need more help?</h3>
                <p className="mt-1 text-sm text-blue-100">Contact our support team for personalized assistance</p>
                <div className="mt-3 space-y-2">
                  <div className="flex items-center text-sm">
                    <Phone className="mr-2 h-4 w-4" />
                    <span>+91 1800-XXX-XXXX</span>
                  </div>
                  <div className="flex items-center text-sm">
                    <Mail className="mr-2 h-4 w-4" />
                    <span>support@votesecure.gov.in</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* FAQs */}
        <div className="md:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle>Frequently Asked Questions</CardTitle>
            </CardHeader>
            <CardContent>
              {filteredFaqs.length > 0 ? (
                <div className="space-y-4">
                  {filteredFaqs.map((faq) => (
                    <div key={faq.id} className="rounded-lg border">
                      <button
                        className="flex w-full items-center justify-between p-4 text-left"
                        onClick={() => setActiveFaq(activeFaq === faq.id ? null : faq.id)}
                      >
                        <h3 className="font-medium text-gray-900">{faq.question}</h3>
                        <ChevronDown
                          className={`h-5 w-5 text-gray-500 transition-transform ${
                            activeFaq === faq.id ? "rotate-180" : ""
                          }`}
                        />
                      </button>
                      {activeFaq === faq.id && (
                        <div className="border-t bg-gray-50 p-4">
                          <p className="text-gray-700">{faq.answer}</p>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="rounded-lg border border-dashed p-6 text-center">
                  <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-gray-100">
                    <Search className="h-6 w-6 text-gray-400" />
                  </div>
                  <h3 className="text-lg font-medium text-gray-900">No results found</h3>
                  <p className="mt-2 text-sm text-gray-500">
                    We couldn't find any FAQs matching your search. Try different keywords or contact our support team.
                  </p>
                </div>
              )}

              <div className="mt-6 rounded-lg bg-blue-50 p-4">
                <div className="flex items-start space-x-3">
                  <Book className="mt-0.5 h-5 w-5 text-blue-600" />
                  <div>
                    <h4 className="text-sm font-medium text-blue-800">User Guide</h4>
                    <p className="mt-1 text-sm text-blue-700">
                      For detailed instructions on using the VoteSecure platform, download our comprehensive user guide.
                    </p>
                    <Button className="mt-3" size="sm">
                      Download User Guide
                      <ArrowRight className="ml-1.5 h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Live Chat */}
          <Card className="mt-6">
            <CardHeader>
              <CardTitle>Live Chat Support</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64 overflow-y-auto rounded-lg border bg-gray-50 p-4">
                {chatMessages.map((message) => (
                  <div
                    key={message.id}
                    className={`mb-3 flex ${message.sender === "user" ? "justify-end" : "justify-start"}`}
                  >
                    <div
                      className={`max-w-[80%] rounded-lg p-3 ${
                        message.sender === "user" ? "bg-blue-600 text-white" : "bg-white text-gray-700 shadow-sm"
                      }`}
                    >
                      <p className="text-sm">{message.message}</p>
                      <p
                        className={`mt-1 text-right text-xs ${
                          message.sender === "user" ? "text-blue-100" : "text-gray-400"
                        }`}
                      >
                        {message.time}
                      </p>
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-4 flex">
                <input
                  type="text"
                  placeholder="Type your message..."
                  value={newMessage}
                  onChange={(e) => setNewMessage(e.target.value)}
                  onKeyPress={(e) => e.key === "Enter" && sendMessage()}
                  className="flex-1 rounded-l-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                />
                <Button
                  className="rounded-l-none rounded-r-md"
                  onClick={sendMessage}
                  disabled={newMessage.trim() === ""}
                >
                  <Send className="mr-1.5 h-4 w-4" />
                  Send
                </Button>
              </div>

              <div className="mt-4 text-center">
                <p className="text-xs text-gray-500">
                  Our support team is available Monday to Friday, 9:00 AM to 6:00 PM.
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

