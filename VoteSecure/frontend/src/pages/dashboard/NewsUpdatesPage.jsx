"use client"

import React, { useState } from "react"
import { Calendar, ChevronRight, Clock, FileText, Filter, MapPin, Search, Shield } from "lucide-react"

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

export default function NewsUpdatesPage() {
  const [searchQuery, setSearchQuery] = useState("")
  const [filterCategory, setFilterCategory] = useState("all")
  const [selectedNews, setSelectedNews] = useState(null)

  // Mock news data
  const newsItems = [
    {
      id: 1,
      title: "Voter List Verification Drive",
      date: "January 10, 2025",
      time: "2 days ago",
      category: "Announcement",
      content:
        "The Election Commission has announced a nationwide voter list verification drive starting next month. All registered voters are requested to verify their details in the electoral roll to ensure accuracy. The verification process can be completed online through the Election Commission's website or by visiting the nearest electoral office.",
      source: "Election Commission of India",
      image: "/placeholder.svg?height=200&width=400",
    },
    {
      id: 2,
      title: "New Voting Machines",
      date: "December 15, 2024",
      time: "1 week ago",
      category: "Technology",
      content:
        "Next-generation voting machines with enhanced security features will be used in the upcoming elections. These machines include improved encryption, paper audit trails, and tamper-detection mechanisms. Training sessions for polling officials on the new machines will be conducted in the coming months.",
      source: "Ministry of Electronics & IT",
      image: "/placeholder.svg?height=200&width=400",
    },
    {
      id: 3,
      title: "Polling Station Changes",
      date: "November 5, 2024",
      time: "2 weeks ago",
      category: "Logistics",
      content:
        "Several polling stations have been relocated to accommodate more voters and improve accessibility. Voters are advised to check their assigned polling station before election day. The updated list of polling stations is available on the Election Commission's website and mobile app.",
      source: "State Election Office",
      image: "/placeholder.svg?height=200&width=400",
    },
    {
      id: 4,
      title: "Voter Awareness Campaign",
      date: "October 20, 2024",
      time: "1 month ago",
      category: "Education",
      content:
        "A nationwide voter awareness campaign has been launched to educate citizens about the importance of voting and the electoral process. The campaign includes workshops, social media outreach, and community events. Special focus is being given to first-time voters and underrepresented communities.",
      source: "National Voter Service Portal",
      image: "/placeholder.svg?height=200&width=400",
    },
    {
      id: 5,
      title: "Election Code of Conduct",
      date: "September 30, 2024",
      time: "2 months ago",
      category: "Regulation",
      content:
        "The Model Code of Conduct for the upcoming General Elections has been published. The code outlines guidelines for political parties, candidates, and government officials during the election period. Violations of the code can result in penalties and disqualification.",
      source: "Election Commission of India",
      image: "/placeholder.svg?height=200&width=400",
    },
  ]

  // Filter news based on search query and filter category
  const filteredNews = newsItems.filter((news) => {
    const matchesSearch =
      news.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      news.content.toLowerCase().includes(searchQuery.toLowerCase())

    if (filterCategory === "all") return matchesSearch
    return matchesSearch && news.category.toLowerCase() === filterCategory.toLowerCase()
  })

  return (
    <div className="p-4 md:p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">News & Updates</h1>
        <p className="text-gray-500">Stay informed about election news and updates</p>
      </div>

      {/* Featured News */}
      <div className="mb-8 overflow-hidden rounded-lg bg-gradient-to-r from-blue-600 to-purple-600 shadow-lg">
        <div className="relative">
          <img
            src="/placeholder.svg?height=300&width=1200"
            alt="Featured News"
            className="h-48 w-full object-cover md:h-64"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent"></div>
          <div className="absolute bottom-0 left-0 p-6">
            <Badge variant="warning" className="mb-2">
              Featured
            </Badge>
            <h2 className="text-2xl font-bold text-white">Election Commission Announces General Elections 2025</h2>
            <p className="mt-2 text-blue-100">
              The Election Commission has announced the schedule for the General Elections 2025, which will be held in
              multiple phases starting April 15.
            </p>
            <div className="mt-4 flex items-center space-x-4">
              <div className="flex items-center text-blue-100">
                <Calendar className="mr-1 h-4 w-4" />
                <span className="text-sm">March 1, 2025</span>
              </div>
              <div className="flex items-center text-blue-100">
                <Clock className="mr-1 h-4 w-4" />
                <span className="text-sm">2 hours ago</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Search and Filter */}
      <div className="mb-6 flex flex-col space-y-4 sm:flex-row sm:items-center sm:justify-between sm:space-y-0">
        <div className="relative w-full sm:max-w-xs">
          <Search className="absolute left-3 top-2.5 h-4 w-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search news..."
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
            <option value="Announcement">Announcements</option>
            <option value="Technology">Technology</option>
            <option value="Logistics">Logistics</option>
            <option value="Education">Education</option>
            <option value="Regulation">Regulations</option>
          </select>
        </div>
      </div>

      {/* News Grid */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {filteredNews.length > 0 ? (
          filteredNews.map((news) => (
            <Card
              key={news.id}
              className="overflow-hidden transition-all hover:shadow-md"
              onClick={() => setSelectedNews(news)}
            >
              <div className="relative h-48">
                <img src={news.image || "/placeholder.svg"} alt={news.title} className="h-full w-full object-cover" />
                <Badge variant="secondary" className="absolute left-3 top-3">
                  {news.category}
                </Badge>
              </div>
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between text-sm text-gray-500">
                  <div className="flex items-center">
                    <Calendar className="mr-1 h-4 w-4" />
                    <span>{news.date}</span>
                  </div>
                  <span>{news.time}</span>
                </div>
                <CardTitle className="mt-2 line-clamp-2">{news.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="line-clamp-3 text-sm text-gray-600">{news.content}</p>
                <div className="mt-4 flex items-center justify-between">
                  <span className="text-xs text-gray-500">Source: {news.source}</span>
                  <button className="flex items-center text-sm font-medium text-blue-600 hover:text-blue-800">
                    Read More
                    <ChevronRight className="ml-1 h-4 w-4" />
                  </button>
                </div>
              </CardContent>
            </Card>
          ))
        ) : (
          <div className="col-span-full rounded-lg border border-gray-200 bg-gray-50 p-6 text-center">
            <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-gray-100">
              <Search className="h-6 w-6 text-gray-400" />
            </div>
            <h3 className="text-lg font-medium text-gray-900">No news found</h3>
            <p className="mt-2 text-sm text-gray-500">
              No news items match your search criteria. Try adjusting your filters.
            </p>
          </div>
        )}
      </div>

      {/* Election Updates */}
      <div className="mt-8">
        <h2 className="mb-4 text-xl font-bold text-gray-900">Election Updates</h2>
        <Card>
          <CardContent className="p-6">
            <div className="space-y-6">
              <div className="flex items-start space-x-4">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-blue-100">
                  <FileText className="h-5 w-5 text-blue-600" />
                </div>
                <div>
                  <h4 className="font-medium text-gray-900">Voter List Publication</h4>
                  <p className="mt-1 text-sm text-gray-600">
                    The final electoral roll for the General Elections 2025 has been published. Voters can check their
                    details on the Election Commission's website.
                  </p>
                  <div className="mt-2 flex items-center text-xs text-gray-500">
                    <Calendar className="mr-1 h-3.5 w-3.5" />
                    <span>February 15, 2025</span>
                  </div>
                </div>
              </div>

              <div className="flex items-start space-x-4">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-green-100">
                  <MapPin className="h-5 w-5 text-green-600" />
                </div>
                <div>
                  <h4 className="font-medium text-gray-900">Polling Station Allocation</h4>
                  <p className="mt-1 text-sm text-gray-600">
                    Polling stations have been allocated to all registered voters. Check your assigned polling station
                    in your voter profile.
                  </p>
                  <div className="mt-2 flex items-center text-xs text-gray-500">
                    <Calendar className="mr-1 h-3.5 w-3.5" />
                    <span>February 10, 2025</span>
                  </div>
                </div>
              </div>

              <div className="flex items-start space-x-4">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-yellow-100">
                  <Shield className="h-5 w-5 text-yellow-600" />
                </div>
                <div>
                  <h4 className="font-medium text-gray-900">Security Measures</h4>
                  <p className="mt-1 text-sm text-gray-600">
                    Enhanced security measures will be implemented at all polling stations to ensure free and fair
                    elections.
                  </p>
                  <div className="mt-2 flex items-center text-xs text-gray-500">
                    <Calendar className="mr-1 h-3.5 w-3.5" />
                    <span>February 5, 2025</span>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* News Modal */}
      {selectedNews && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
          <div className="w-full max-w-3xl rounded-lg bg-white p-6">
            <div className="mb-4 flex items-center justify-between">
              <Badge variant="secondary">{selectedNews.category}</Badge>
              <button
                className="rounded-full p-1 text-gray-500 hover:bg-gray-100 hover:text-gray-700"
                onClick={() => setSelectedNews(null)}
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            <h2 className="text-2xl font-bold text-gray-900">{selectedNews.title}</h2>

            <div className="mt-2 flex items-center space-x-4 text-sm text-gray-500">
              <div className="flex items-center">
                <Calendar className="mr-1 h-4 w-4" />
                <span>{selectedNews.date}</span>
              </div>
              <div>
                <span>Source: {selectedNews.source}</span>
              </div>
            </div>

            <div className="mt-4">
              <img
                src={selectedNews.image || "/placeholder.svg"}
                alt={selectedNews.title}
                className="h-64 w-full rounded-lg object-cover"
              />
            </div>

            <div className="mt-4 text-gray-700">
              <p className="whitespace-pre-line">{selectedNews.content}</p>
              <p className="mt-4">
                Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam auctor, nisl eget ultricies tincidunt,
                nisl nisl aliquam nisl, eget ultricies nisl nisl eget nisl. Nullam auctor, nisl eget ultricies
                tincidunt, nisl nisl aliquam nisl, eget ultricies nisl nisl eget nisl.
              </p>
              <p className="mt-4">
                Nullam auctor, nisl eget ultricies tincidunt, nisl nisl aliquam nisl, eget ultricies nisl nisl eget
                nisl. Nullam auctor, nisl eget ultricies tincidunt, nisl nisl aliquam nisl, eget ultricies nisl nisl
                eget nisl.
              </p>
            </div>

            <div className="mt-6 flex justify-end">
              <button
                className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
                onClick={() => setSelectedNews(null)}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

