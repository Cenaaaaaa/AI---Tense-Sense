"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"

type TenseType = "Present Tense" | "Past Tense" | "Future Tense" | null

interface PredictionResult {
  tense: TenseType
  confidence?: number
}

export default function Home() {
  const [sentence, setSentence] = useState("")
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  const handlePredict = async () => {
    if (!sentence.trim()) {
      setError("Please enter a sentence")
      return
    }

    setLoading(true)
    setError("")
    setResult(null)

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ sentence }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(
          errorData.error ||
            "Failed to predict tense.",
        )
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "An error occurred.",
      )
    } finally {
      setLoading(false)
    }
  }

  const getTenseColor = (tense: TenseType) => {
    switch (tense) {
      case "Present Tense":
        return "bg-blue-100 text-blue-900 border-blue-300"
      case "Past Tense":
        return "bg-amber-100 text-amber-900 border-amber-300"
      case "Future Tense":
        return "bg-green-100 text-green-900 border-green-300"
      default:
        return ""
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-4 md:p-8">
      <div className="max-w-2xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-slate-900 mb-2">Tense Sense</h1>
          <p className="text-slate-600">Detect whether a sentence is in past, present, or future tense</p>
        </div>

        {/* Main Card */}
        <Card className="shadow-lg">
          <CardHeader className="bg-slate-50 border-b">
            <CardTitle className="text-slate-900">Analyze Your Sentence</CardTitle>
            <CardDescription>Enter any English sentence to classify its tense</CardDescription>
          </CardHeader>
          <CardContent className="pt-6 space-y-6">
            {/* Input Section */}
            <div className="space-y-3">
              <label htmlFor="sentence" className="block text-sm font-medium text-slate-700">
                Enter a sentence:
              </label>
              <Textarea
                id="sentence"
                placeholder="Type or paste your English sentence here... e.g., 'She will travel to Paris next week'"
                value={sentence}
                onChange={(e) => {
                  setSentence(e.target.value)
                  setError("")
                }}
                className="min-h-24 resize-none"
              />
              <Button onClick={handlePredict} disabled={loading} className="w-full bg-slate-900 hover:bg-slate-800">
                {loading ? "Analyzing..." : "Predict Tense"}
              </Button>
            </div>

            {/* Error Message */}
            {error && (
              <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-red-700 text-sm">{error}</p>
              </div>
            )}

            {/* Result Section */}
            {result && (
              <div className="p-6 bg-slate-50 rounded-lg border-2 border-slate-200">
                <p className="text-sm text-slate-600 mb-2">Predicted Tense:</p>
                <div
                  className={`inline-block px-4 py-2 rounded-lg border-2 font-semibold ${getTenseColor(result.tense)}`}
                >
                  {result.tense}
                </div>
              </div>
            )}

            {/* Examples Section */}
            <div className="mt-8 pt-6 border-t">
              <h3 className="font-semibold text-slate-900 mb-4">Try these examples:</h3>
              <div className="space-y-2">
                <button
                  onClick={() => setSentence("She sings beautifully")}
                  className="block w-full text-left p-3 rounded-lg bg-slate-100 hover:bg-blue-100 transition text-slate-700 hover:text-blue-900 text-sm"
                >
                  📍 She sings beautifully <span className="float-right text-blue-600">Present</span>
                </button>
                <button
                  onClick={() => setSentence("They discovered a new species")}
                  className="block w-full text-left p-3 rounded-lg bg-slate-100 hover:bg-amber-100 transition text-slate-700 hover:text-amber-900 text-sm"
                >
                  📍 They discovered a new species <span className="float-right text-amber-600">Past</span>
                </button>
                <button
                  onClick={() => setSentence("We will finish this project by tomorrow")}
                  className="block w-full text-left p-3 rounded-lg bg-slate-100 hover:bg-green-100 transition text-slate-700 hover:text-green-900 text-sm"
                >
                  📍 We will finish this project by tomorrow <span className="float-right text-green-600">Future</span>
                </button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Footer */}
        <p className="text-center text-sm text-slate-500 mt-6">Powered by TF-IDF and Logistic Regression</p>
      </div>
    </main>
  )
}
