1. Converts all dates into this format: YYYY-MM-DD HH:MM:SS
2. Replace all instances of "elements" with "traces" through out the UI. General rule: make sure you're calling them traces not elements.
3. Dashboard: 
    - Unique traces number should be at the top and total traces number should be at the bottom.
    - Remove (With Tool Calls) and replace it with "With Metadata"
    - Remove (Time Span) and include it in "Analyzing {total_traces} traces ({unique_traces} unique traces) from {time_period_start} to {time_period_end} {time_span_days} days". Also highlight the time period in some color also the time span days.
    - Remove "Performance Metrics"
    - Make "model distribution" a pie chart and also "use case distribution" a pie chart (side by side).
    - Remove "Multi-element stats panel"
4. Add appropriate filters in /errors page
5. In /traces add filters for "With Metadata", "Multi-turn"