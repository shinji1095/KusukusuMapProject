import os
import logging
from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from .api import get_route_with_crosswalk
from .utils import parse_latlng_pair

logger = logging.getLogger(__name__)

@csrf_exempt
def route_crosswalk_view(request):
    origin = parse_latlng_pair(request.GET.get("origin"))
    destination = parse_latlng_pair(request.GET.get("destination"))
    api_key = os.getenv("GOOGLE_API_KEY")

    if not origin or not destination:
        return HttpResponseBadRequest("Missing origin or destination")

    try:
        result, log_dir =  get_route_with_crosswalk(
            origin, 
            destination, 
            api_key, 
            cache_dir="cache",
            sample_step=5, 
            search_radius=200, 
            target_radius=50,
            cluster_eps=30, 
            visualize=(request.method == "GET"),
            debug=True
        )

        # POST: 
        if request.method == "POST":
            total_duration = result.get("total_duration", 0)
            matched_crosswalks = result.get("matched_crosswalks", [])
            route_poly = result.get("route_polyline", "")

            json_result = {
                "origin": origin,
                "destination": destination,
                "total_duration": total_duration,
                "num_crosswalks": len(matched_crosswalks),
                "route_polyline": route_poly,
                "matched_crosswalks": matched_crosswalks,
            }

            return JsonResponse(json_result)

        # GET: 
        elif request.method == "GET":
            html_path = os.path.join(log_dir, "debug_route.html")
            try:
                with open(html_path, "r", encoding="utf-8") as f:
                    html = f.read()
                return HttpResponse(html)
            except FileNotFoundError:
                return JsonResponse({"error": "debug_route.html not found in log_dir"}, status=500)

    except Exception as e:
        logger.exception("Internal error during route_crosswalk_view")
        return JsonResponse({"error": str(e)}, status=500)
