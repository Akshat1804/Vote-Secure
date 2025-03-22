// Mock candidates data
export const getCandidates = () => {
    return [
      {
        id: 1,
        name: "Rajesh Kumar",
        party: "National Democratic Alliance",
        partyShort: "NDA",
        symbol: "🌷", // Lotus symbol
        image: "/placeholder.svg?height=100&width=100",
        color: "bg-orange-100 border-orange-300",
        buttonColor: "bg-indigo-600 hover:bg-indigo-700",
        category: "national",
        details: {
          age: 52,
          education: "Ph.D. in Public Policy",
          experience: "15 years as Member of Parliament",
          achievements: [
            "Led infrastructure development projects worth ₹500 crore",
            "Authored 3 major policy reforms in education sector",
            "Increased constituency development index by 25%",
          ],
          manifesto: [
            "Infrastructure development in rural areas",
            "Job creation through manufacturing sector",
            "National security enhancement",
            "Digital India initiatives",
          ],
          socialMedia: {
            twitter: "@rajeshkumar",
            facebook: "rajeshkumarofficial",
            instagram: "rajesh.kumar.official",
          },
        },
      },
      {
        id: 2,
        name: "Priya Sharma",
        party: "United Progressive Alliance",
        partyShort: "UPA",
        symbol: "✋", // Hand symbol
        image: "/placeholder.svg?height=100&width=100",
        color: "bg-blue-100 border-blue-300",
        buttonColor: "bg-indigo-600 hover:bg-indigo-700",
        category: "national",
        details: {
          age: 48,
          education: "Master's in Economics",
          experience: "Former State Minister, 10 years in politics",
          achievements: [
            "Implemented universal healthcare program in her state",
            "Reduced gender gap in education by 15%",
            "Established 50 women's empowerment centers",
          ],
          manifesto: [
            "Universal healthcare access",
            "Education reforms and scholarships",
            "Women's empowerment programs",
            "Environmental protection policies",
          ],
          socialMedia: {
            twitter: "@priyasharma",
            facebook: "priyasharmaofficial",
            instagram: "priya.sharma.official",
          },
        },
      },
      {
        id: 3,
        name: "Amit Patel",
        party: "Regional People's Front",
        partyShort: "RPF",
        symbol: "🚲", // Bicycle symbol
        image: "/placeholder.svg?height=100&width=100",
        color: "bg-green-100 border-green-300",
        buttonColor: "bg-indigo-600 hover:bg-indigo-700",
        category: "regional",
        details: {
          age: 45,
          education: "Law Degree",
          experience: "Grassroots activist, 8 years as MLA",
          achievements: [
            "Secured water rights for 200 villages",
            "Led farmers' movement for fair crop prices",
            "Established 25 rural development centers",
          ],
          manifesto: [
            "Regional autonomy and development",
            "Agricultural subsidies and farmer support",
            "Local language and cultural preservation",
            "Water resource management",
          ],
          socialMedia: {
            twitter: "@amitpatel",
            facebook: "amitpatelofficial",
            instagram: "amit.patel.official",
          },
        },
      },
      {
        id: 4,
        name: "Sunita Reddy",
        party: "Progressive Democratic Party",
        partyShort: "PDP",
        symbol: "🔔", // Bell symbol
        image: "/placeholder.svg?height=100&width=100",
        color: "bg-purple-100 border-purple-300",
        buttonColor: "bg-indigo-600 hover:bg-indigo-700",
        category: "regional",
        details: {
          age: 39,
          education: "MBA and Social Work",
          experience: "NGO founder, 5 years in politics",
          achievements: [
            "Launched tech education program reaching 10,000 students",
            "Created urban housing initiative for 5,000 families",
            "Established innovation hub creating 2,000 jobs",
          ],
          manifesto: [
            "Universal basic income pilot programs",
            "Technology sector development",
            "Healthcare modernization",
            "Urban housing and transportation",
          ],
          socialMedia: {
            twitter: "@sunitareddy",
            facebook: "sunitareddyofficial",
            instagram: "sunita.reddy.official",
          },
        },
      },
      {
        id: 5,
        name: "Vikram Singh",
        party: "People's Democratic Front",
        partyShort: "PDF",
        symbol: "🌟", // Star symbol
        image: "/placeholder.svg?height=100&width=100",
        color: "bg-yellow-100 border-yellow-300",
        buttonColor: "bg-indigo-600 hover:bg-indigo-700",
        category: "independent",
        details: {
          age: 41,
          education: "Civil Engineering & Public Administration",
          experience: "Civil servant for 12 years, Independent candidate",
          achievements: [
            "Led urban renewal projects in 3 major cities",
            "Implemented transparent governance systems",
            "Reduced bureaucratic processes by 30%",
          ],
          manifesto: [
            "Administrative reforms and transparency",
            "Smart city initiatives",
            "Youth employment programs",
            "Anti-corruption measures",
          ],
          socialMedia: {
            twitter: "@vikramsingh",
            facebook: "vikramsinghofficial",
            instagram: "vikram.singh.official",
          },
        },
      },
      {
        id: 6,
        name: "Meera Desai",
        party: "Independent",
        partyShort: "IND",
        symbol: "📝", // Pencil symbol
        image: "/placeholder.svg?height=100&width=100",
        color: "bg-gray-100 border-gray-300",
        buttonColor: "bg-indigo-600 hover:bg-indigo-700",
        category: "independent",
        details: {
          age: 36,
          education: "Journalism and Political Science",
          experience: "Journalist, Social Activist, First-time candidate",
          achievements: [
            "Award-winning investigative journalist",
            "Led citizen movements for environmental protection",
            "Founded education initiative for underprivileged children",
          ],
          manifesto: [
            "Media freedom and right to information",
            "Environmental sustainability",
            "Education for all",
            "Grassroots democracy strengthening",
          ],
          socialMedia: {
            twitter: "@meeradesai",
            facebook: "meeradesaiofficial",
            instagram: "meera.desai.official",
          },
        },
      },
    ]
  }
  
  // Generate a vote receipt
  export const generateVoteReceipt = (userData) => {
    return {
      id: Math.random().toString(36).substring(2, 15),
      timestamp: new Date().toISOString(),
      constituency: userData?.constituency || "Central Delhi",
      boothId: "B" + Math.floor(1000 + Math.random() * 9000),
      verificationCode: Math.random().toString(36).substring(2, 10).toUpperCase(),
      electionId: "GE2025-" + Math.floor(10000 + Math.random() * 90000),
    }
  }
  
  